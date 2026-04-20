import os
import json
import re
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


# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are a scheduling assistant.

Extract ALL scheduling tasks from user input.

STRICT RULES:
- ALWAYS return valid JSON
- DO NOT include explanations
- DO NOT include text outside JSON

OUTPUT:
{
  "active_window": {},
  "tasks": [],
  "do_not_disturb": [],
  "exclusions": []
}
"""


# =========================
# HELPERS
# =========================
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


# =========================
# SAFE JSON
# =========================
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


# =========================
# ACTIVE WINDOW FIX
# =========================
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


# =========================
# NORMALIZE TASKS
# =========================
def normalize_tasks(parsed):
    tasks = parsed.get("tasks", [])
    result = []

    for t in tasks:
        if not isinstance(t, dict):
            continue

        result.append({
            "type": t.get("type"),
            "enabled": bool(t.get("enabled", True)),
            "interval_minutes": safe_int(t.get("interval_minutes")),
            "duration_seconds": safe_int(t.get("duration_seconds")),
            "start_time": t.get("start_time"),
            "end_time": t.get("end_time")
        })

    return result


# =========================
# SCHEDULER
# =========================
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

        current = start + interval

        while current <= end:
            raw.append({
                "time_min": current,
                "task": t.get("type"),
                "priority": PRIORITY.get(t.get("type"), 99)
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
                final.append({**e, "time_min": new_time})
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


# =========================
# DEVICE SCHEMA
# =========================
def convert(parsed):

    tasks = normalize_tasks(parsed)
    active = parsed.get("active_window", {})
    dnd_list = parsed.get("do_not_disturb", [])

    sh, sm = parse_time(active.get("start"))
    eh, em = parse_time(active.get("end"))

    hydration = next((t for t in tasks if t["type"] == "hydration"), {})
    eye = next((t for t in tasks if t["type"] == "eye"), {})
    stretch = next((t for t in tasks if t["type"] == "stretch"), {})
    walk = next((t for t in tasks if t["type"] == "walk"), {})

    dnd = {"enabled": False, "sh": 0, "sm": 0, "eh": 0, "em": 0}
    if dnd_list:
        sh_d, sm_d = parse_time(dnd_list[0].get("start"))
        eh_d, em_d = parse_time(dnd_list[0].get("end"))
        dnd.update({"enabled": True, "sh": sh_d, "sm": sm_d, "eh": eh_d, "em": em_d})

    return {
        "hydration": {
            "enabled": hydration.get("enabled", False),
            "interval_ms": (hydration.get("interval_minutes") or 30)*60000,
            "start_hour": sh,
            "start_min": sm,
            "end_hour": eh,
            "end_min": em
        },
        "eye": {
            "enabled": eye.get("enabled", False),
            "interval_ms": (eye.get("interval_minutes") or 20)*60000,
            "duration_ms": (eye.get("duration_seconds") or 20)*1000
        },
        "stretch": {
            "enabled": stretch.get("enabled", False),
            "interval_ms": (stretch.get("interval_minutes") or 60)*60000
        },
        "walk": {
            "enabled": walk.get("enabled", False),
            "interval_ms": (walk.get("interval_minutes") or 60)*60000
        },
        "dnd": dnd,
        "schedule": generate_schedule(tasks, (sh, sm), (eh, em), dnd)
    }


# =========================
# ROUTE
# =========================
@app.route("/parse", methods=["POST"])
def parse_schedule():
    try:
        logs = []

        def log(msg):
            logs.append(f"{datetime.now(IST).strftime('%H:%M:%S')} - {msg}")

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

        return jsonify({"data": final, "logs": logs})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(port=5000)