import os
import json
import re
import time
from datetime import datetime
import pytz
from flask_cors import CORS
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
app.json.sort_keys = False

CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
IST = pytz.timezone('Asia/Kolkata')


SYSTEM_PROMPT = """
You are an intelligent scheduling parser.

Your job is to convert natural language into STRICT structured JSON.

-----------------------------------
CORE REQUIREMENTS
-----------------------------------

You MUST extract:

1. Active working window
2. Tasks (hydration, eye, stretch, walk)
3. Do not disturb periods
4. Exclusions (if any)

-----------------------------------
SUPPORTED TASK TYPES
-----------------------------------

You MUST ONLY use these exact types:

- "hydration"
- "eye"
- "stretch"
- "walk"

NEVER invent new types.
NEVER leave type empty.

-----------------------------------
TASK RULES
-----------------------------------

Each task MUST follow this format:

{
  "type": "hydration | eye | stretch | walk",
  "enabled": true,
  "interval_minutes": number,
  "duration_seconds": number (ONLY for eye),
  "start_time": null,
  "end_time": null
}

Rules:
- If a task is mentioned → enabled MUST be true
- If not mentioned → DO NOT include it
- hydration/stretch/walk → only interval_minutes
- eye → must include BOTH interval_minutes AND duration_seconds (default 20 sec if not specified)

-----------------------------------
ACTIVE WINDOW RULES
-----------------------------------

Extract working hours like:

"9 to 5" → 
{
  "start": "09:00",
  "end": "17:00"
}

If not provided → leave empty {}

-----------------------------------
DO NOT DISTURB RULES
-----------------------------------

Example:
"avoid 1 to 2" →

[
  {
    "start": "13:00",
    "end": "14:00"
  }
]

-----------------------------------
STRICT OUTPUT FORMAT
-----------------------------------

You MUST return ONLY valid JSON.

NO explanations  
NO text outside JSON  
NO comments  

-----------------------------------
EXAMPLE
-----------------------------------

Input:
"I work 9 to 5, drink water every 30 min, eye breaks every 20 min"

Output:
{
  "active_window": {
    "start": "09:00",
    "end": "17:00"
  },
  "tasks": [
    {
      "type": "hydration",
      "enabled": true,
      "interval_minutes": 30,
      "duration_seconds": null,
      "start_time": null,
      "end_time": null
    },
    {
      "type": "eye",
      "enabled": true,
      "interval_minutes": 20,
      "duration_seconds": 20,
      "start_time": null,
      "end_time": null
    }
  ],
  "do_not_disturb": [],
  "exclusions": []
}
"""


def safe_int(val, default=None):
    try:
        return int(val)
    except:
        return default


def parse_time(t):
    try:
        if not t:
            return 0, 0
        h, m = t.split(":")
        return int(h), int(m)
    except:
        return 0, 0


def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass

    return {
        "active_window": {},
        "tasks": [],
        "do_not_disturb": [],
        "exclusions": []
    }


def ensure_defaults(parsed):
    parsed.setdefault("tasks", [])
    parsed.setdefault("active_window", {})
    parsed.setdefault("do_not_disturb", [])
    parsed.setdefault("exclusions", [])
    return parsed


def infer_active_window(text, parsed):
    if parsed.get("active_window", {}).get("start"):
        return parsed

    text = text.lower()

    match = re.search(r"(\d{1,2})\s*(am|pm)?\s*(to|-)\s*(\d{1,2})\s*(am|pm)?", text)

    if match:
        sh = int(match.group(1))
        eh = int(match.group(4))

        if match.group(2) == "pm" and sh != 12:
            sh += 12
        if match.group(5) == "pm" and eh != 12:
            eh += 12

        parsed["active_window"] = {
            "start": f"{sh:02d}:00",
            "end": f"{eh:02d}:00"
        }

    return parsed


def normalize_tasks(parsed):
    tasks = parsed.get("tasks", [])
    result = []

    for t in tasks:
        if not isinstance(t, dict):
            continue

        task_type = t.get("type")

        # 🔥 FIX: skip invalid tasks
        if not task_type:
            continue

        result.append({
            "type": task_type,
            "enabled": bool(t.get("enabled", True)),
            "interval_minutes": safe_int(t.get("interval_minutes")),
            "duration_seconds": safe_int(t.get("duration_seconds")),
            "start_time": t.get("start_time"),
            "end_time": t.get("end_time")
        })

    return result


PRIORITY = {"hydration": 1, "eye": 2, "stretch": 3, "walk": 3}
MIN_GAP = 5


def is_in_dnd(time_min, dnd):
    if not dnd["enabled"]:
        return False

    start = dnd["sh"] * 60 + dnd["sm"]
    end = dnd["eh"] * 60 + dnd["em"]

    return start <= time_min <= end


def generate_schedule(tasks, global_start, global_end, dnd):
    raw = []

    start = global_start[0]*60 + global_start[1]
    end = global_end[0]*60 + global_end[1]

    for t in tasks:
        if not t.get("enabled"):
            continue

        interval = t.get("interval_minutes")
        if not interval:
            continue

        task_type = t.get("type")

        # 🔥 EXTRA SAFETY
        if not task_type:
            continue

        current = start + interval

        while current <= end:
            raw.append({
                "time_min": current,
                "task": task_type,
                "priority": PRIORITY.get(task_type, 99)
            })
            current += interval

    filtered = [e for e in raw if not is_in_dnd(e["time_min"], dnd)]
    filtered.sort(key=lambda x: (x["time_min"], x["priority"]))

    final = []

    for e in filtered:
        if not final:
            final.append(e)
            continue

        last = final[-1]

        if e["time_min"] - last["time_min"] < MIN_GAP:
            new_time = last["time_min"] + MIN_GAP
            if new_time <= end:
                final.append({
                    "time_min": new_time,
                    "task": e["task"],
                    "priority": e["priority"]
                })
        else:
            final.append(e)

    schedule = []
    for e in final:
        h = e["time_min"] // 60
        m = e["time_min"] % 60

        schedule.append({
            "time": f"{h:02d}:{m:02d}",
            "task": e["task"]
        })

    return schedule


def convert(parsed):
    tasks = normalize_tasks(parsed)

    active = parsed.get("active_window", {})
    dnd_list = parsed.get("do_not_disturb", [])

    sh, sm = parse_time(active.get("start"))
    eh, em = parse_time(active.get("end"))

    hydration = next((t for t in tasks if t["type"] == "hydration"), None)
    eye = next((t for t in tasks if t["type"] == "eye"), None)
    stretch = next((t for t in tasks if t["type"] == "stretch"), None)
    walk = next((t for t in tasks if t["type"] == "walk"), None)

    hydration_enabled = hydration.get("enabled", True) if hydration else False
    eye_enabled = eye.get("enabled", True) if eye else False
    stretch_enabled = stretch.get("enabled", True) if stretch else False
    walk_enabled = walk.get("enabled", True) if walk else False

    dnd = {"enabled": False, "sh": 0, "sm": 0, "eh": 0, "em": 0}
    if dnd_list:
        sh_d, sm_d = parse_time(dnd_list[0].get("start"))
        eh_d, em_d = parse_time(dnd_list[0].get("end"))
        dnd.update({"enabled": True, "sh": sh_d, "sm": sm_d, "eh": eh_d, "em": em_d})

    return {
        "hydration": {
            "enabled": hydration_enabled,
            "interval_ms": (hydration.get("interval_minutes") if hydration else 30) * 60000,
            "start_hour": sh,
            "start_min": sm,
            "end_hour": eh,
            "end_min": em
        },
        "eye": {
            "enabled": eye_enabled,
            "interval_ms": (eye.get("interval_minutes") if eye else 20) * 60000,
            "duration_ms": (eye.get("duration_seconds") if eye else 20) * 1000
        },
        "stretch": {
            "enabled": stretch_enabled,
            "interval_ms": (stretch.get("interval_minutes") if stretch else 60) * 60000
        },
        "walk": {
            "enabled": walk_enabled,
            "interval_ms": (walk.get("interval_minutes") if walk else 60) * 60000
        },
        "dnd": dnd,
        "schedule": generate_schedule(tasks, (sh, sm), (eh, em), dnd)
    }


@app.route("/parse", methods=["POST"])
def parse_schedule():
    start_time = time.time()
    logs = []

    def log(msg):
        logs.append(f"{datetime.now(IST).strftime('%H:%M:%S')} - {msg}")

    try:
        data = request.get_json()
        user_text = data.get("text")

        log("Request received")
        log("Calling OpenAI")

        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
        )

        log("Model response received")

        raw = response.output_text

        parsed = safe_json_parse(raw)
        log("JSON parsed safely")

        parsed = ensure_defaults(parsed)
        parsed = infer_active_window(user_text, parsed)
        log("Defaults + active window applied")

        final = convert(parsed)
        log("Converted to schema")

        elapsed = round(time.time() - start_time, 2)
        log(f"Total elapsed: {elapsed}s")

        return jsonify({"data": final, "logs": logs})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(port=5000)