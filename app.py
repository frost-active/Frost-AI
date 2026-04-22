import os
import json
from datetime import datetime
import pytz
from flask_cors import CORS
from flask import Flask, request, jsonify
from openai import OpenAI
import re

app = Flask(__name__)
app.json.sort_keys = False

CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

IST = pytz.timezone("Asia/Kolkata")

with open("frost-config.json", "r") as f:
    BASE_SCHEMA = json.load(f)

SYSTEM_PROMPT = """
You are a strict scheduling extraction engine.

Return ONLY valid JSON. No explanations.

Schema:
{
  "active_window": { "start": "HH:MM", "end": "HH:MM" },
  "tasks": [],
  "do_not_disturb": [],
  "exclusions": []
}

Rules:
- Use 24-hour format HH:MM
- Extract ONLY what user explicitly says
- DO NOT hallucinate tasks
- DO NOT assume defaults unless task is mentioned
- Convert AM/PM to 24-hour format

Allowed tasks:
- hydration
- eye
- stretch
- walk

Each task:
{
  "type": "hydration | eye | stretch | walk",
  "enabled": true,
  "interval_minutes": number,
  "duration_seconds": number or null,
  "start_time": "HH:MM" or null,
  "end_time": "HH:MM" or null
}

Defaults (only if mentioned):
- hydration: 30 min
- eye: 20 min (20 sec duration)
- stretch: 60 min
- walk: 90 min

DND:
Extract phrases like:
- avoid notifications
- do not disturb
- no reminders

Format:
{
  "start": "HH:MM",
  "end": "HH:MM"
}

Return ONLY JSON.
"""

def safe_int(v):
    try: return int(v)
    except: return None

def parse_time(t):
    try:
        if not t: return 0,0
        t = t.strip().lower()
        m = re.match(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', t)
        if m:
            h = int(m.group(1))
            mnt = int(m.group(2)) if m.group(2) else 0
            if m.group(3)=="pm" and h!=12: h+=12
            if m.group(3)=="am" and h==12: h=0
            return h,mnt
        h,mnt = t.split(":")
        return int(h),int(mnt)
    except:
        return 0,0

def extract_dnd(text):
    m = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(to|-)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', text.lower())
    if not m: return None

    def cv(h,mn,p):
        h=int(h); mn=int(mn) if mn else 0
        if p=="pm" and h!=12: h+=12
        if p=="am" and h==12: h=0
        return h,mn

    sh=cv(m.group(1),m.group(2),m.group(3))
    eh=cv(m.group(5),m.group(6),m.group(7))
    return {"sh":sh[0],"sm":sh[1],"eh":eh[0],"em":eh[1]}

def normalize(parsed):
    tasks = parsed.get("tasks")
    if not isinstance(tasks,list): return []
    out=[]
    for t in tasks:
        if not isinstance(t,dict): continue
        out.append({
            "type":t.get("type"),
            "enabled":t.get("enabled") is True,
            "interval":safe_int(t.get("interval_minutes")),
            "duration":safe_int(t.get("duration_seconds"))
        })
    return out

def schedule(tasks,start,end,dnd):
    res=[]
    s=start[0]*60+start[1]
    e=end[0]*60+end[1]

    for t in tasks:
        if not t["enabled"]: continue
        iv=t["interval"]
        if iv is None:
            if t["type"]=="hydration": iv=30
            elif t["type"]=="eye": iv=20
            elif t["type"]=="stretch": iv=60
            elif t["type"]=="walk": iv=90
            else: continue

        cur=s+iv
        while cur<=e:
            blocked=False
            if dnd["enabled"]:
                ds=dnd["sh"]*60+dnd["sm"]
                de=dnd["eh"]*60+dnd["em"]
                if ds<=de: blocked=ds<=cur<=de
                else: blocked=cur>=ds or cur<=de

            if not blocked:
                res.append({"time":f"{cur//60:02d}:{cur%60:02d}","task":t["type"]})

            cur+=iv
    return res

def convert(parsed,text):
    cfg=json.loads(json.dumps(BASE_SCHEMA))

    tasks=normalize(parsed)
    tasks=[t for t in tasks if t["enabled"]]

    aw=parsed.get("active_window") or {}
    sh,sm=parse_time(aw.get("start"))
    eh,em=parse_time(aw.get("end"))

    if sh==0 and eh==0 and tasks:
        sh,sm=9,0; eh,em=17,0

    h=next((t for t in tasks if t["type"]=="hydration"),{})
    e=next((t for t in tasks if t["type"]=="eye"),{})
    s=next((t for t in tasks if t["type"]=="stretch"),{})
    w=next((t for t in tasks if t["type"]=="walk"),{})

    cfg["_meta"]["ts_written"]=int(datetime.now().timestamp())

    cfg["hydration"]["enabled"]=h.get("enabled",False)
    cfg["hydration"]["interval_ms"]=(h.get("interval") or 30)*60000
    cfg["hydration"]["start_hour"]=sh
    cfg["hydration"]["start_min"]=sm
    cfg["hydration"]["end_hour"]=eh
    cfg["hydration"]["end_min"]=em

    cfg["eye"]["enabled"]=e.get("enabled",False)
    cfg["eye"]["interval_ms"]=(e.get("interval") or 20)*60000
    cfg["eye"]["start_hour"]=sh
    cfg["eye"]["start_min"]=sm
    cfg["eye"]["end_hour"]=eh
    cfg["eye"]["end_min"]=em

    cfg["stretch"]["enabled"]=s.get("enabled",False)
    cfg["stretch"]["interval_ms"]=(s.get("interval") or 60)*60000
    cfg["stretch"]["start_hour"]=sh
    cfg["stretch"]["start_min"]=sm
    cfg["stretch"]["end_hour"]=eh
    cfg["stretch"]["end_min"]=em

    cfg["walk"]["enabled"]=w.get("enabled",False)
    cfg["walk"]["interval_min"]=(w.get("interval") or 60)

    dnd={"enabled":False,"sh":0,"sm":0,"eh":0,"em":0}
    fallback=extract_dnd(text)
    if fallback:
        dnd.update({"enabled":True,**fallback})

    cfg["dnd"].update(dnd)

    cfg["schedule"]=schedule(tasks,(sh,sm),(eh,em),dnd)

    return cfg

@app.route("/parse",methods=["POST"])
def parse():
    try:
        data=request.get_json()
        text=data.get("text")

        res=client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":text}
            ]
        )

        parsed=json.loads(res.output_text)
        return jsonify({"data":convert(parsed,text)})

    except Exception as e:
        return jsonify({"error":str(e)}),400

if __name__=="__main__":
    app.run(port=5000)