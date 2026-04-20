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

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

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
- If unsure, return best-effort JSON

SUPPORTED TASKS:
- hydration
- eye
- stretch
- walk

ALSO EXTRACT:
"active_window": {
  "start": "HH:MM",
  "end": "HH:MM"
}

OUTPUT:
{
  "active_window": {},
  "tasks": [],
  "do_not_disturb": [],
  "exclusions": []
}
"""


# =========================
# SAFE HELPERS
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
# 🧠 RESILIENT JSON PARSER
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
    if "tasks" not in parsed or not isinstance(parsed["tasks"], list):
        parsed["tasks"] = []

    if "active_window" not in parsed:
        parsed["active_window"] = {}

    if "do_not_disturb" not in parsed:
        parsed["do_not_disturb"] = []

    if "exclusions" not in parsed:
        parsed["exclusions"] = []

    return parsed


# =========================
# NORMALIZE TASKS
# =========================
def normalize_tasks(parsed):
    tasks = parsed.get("tasks", [])

    normalized = []

    for t in tasks:
        if not isinstance(t, dict):
            continue

        normalized.append({
            "type": t.get("type"),
            "enabled": bool(t.get("enabled", True)),
            "interval_minutes": safe_int(t.get("interval_minutes")),
            "duration_seconds": safe_int(t.get("duration_seconds")),
            "start_time": t.get("start_time"),
            "end_time": t.get("end_time")
        })

    return normalized


# =========================
# SCHEDULER (UNCHANGED)
# =========================
PRIORITY = {
    "hydration": 1,
    "eye": 2,
    "stretch": 3,
    "walk": 3
}

MIN_GAP = 5


def is_in_dnd(time_min, dnd):
    if not dnd["enabled"]:
        return False

    start = dnd["sh"] * 60 + dnd["sm"]
    end = dnd["eh"] * 60 + dnd["em"]

    return start <= time_min <= end


def generate_schedule(tasks, global_start, global_end, dnd):
    raw_events = []

    start_minutes = global_start[0] * 60 + global_start[1]
    end_minutes = global_end[0] * 60 + global_end[1]

    for t in tasks:
        if not t.get("enabled"):
            continue

        interval = t.get("interval_minutes")
        if not interval:
            continue

        current = start_minutes + interval

        while current <= end_minutes:
            raw_events.append({
                "time_min": current,
                "task": t.get("type"),
                "priority": PRIORITY.get(t.get("type"), 99)
            })
            current += interval

    filtered = [
        e for e in raw_events
        if not is_in_dnd(e["time_min"], dnd)
    ]

    filtered.sort(key=lambda x: (x["time_min"], x["priority"]))

    final = []

    for event in filtered:
        if not final:
            final.append(event)
            continue

        last = final[-1]

        if event["time_min"] - last["time_min"] < MIN_GAP:
            new_time = last["time_min"] + MIN_GAP

            if new_time <= end_minutes:
                final.append({
                    "time_min": new_time,
                    "task": event["task"],
                    "priority": event["priority"]
                })
        else:
            final.append(event)

    schedule = []
    for e in final:
        hour = e["time_min"] // 60
        minute = e["time_min"] % 60

        schedule.append({
            "time": f"{str(hour).zfill(2)}:{str(minute).zfill(2)}",
            "task": e["task"]
        })

    return schedule


# =========================
# DEVICE SCHEMA (UNCHANGED)
# =========================
def convert_to_device_schema(parsed):

    tasks = normalize_tasks(parsed)

    active = parsed.get("active_window") or {}
    dnd_list = parsed.get("do_not_disturb") or []
    exclusions = parsed.get("exclusions") or []

    global_sh, global_sm = parse_time(active.get("start"))
    global_eh, global_em = parse_time(active.get("end"))

    hydration = {}
    eye = {}
    stretch = {}
    walk = {}

    for t in tasks:
        if t["type"] == "hydration":
            hydration = t
        elif t["type"] == "eye":
            eye = t
        elif t["type"] == "stretch":
            stretch = t
        elif t["type"] == "walk":
            walk = t

    # HYDRATION
    h_enabled = hydration.get("enabled", False)
    h_interval = (hydration.get("interval_minutes") or 30) * 60 * 1000

    sh, sm = parse_time(hydration.get("start_time")) if hydration.get("start_time") else (global_sh, global_sm)
    eh, em = parse_time(hydration.get("end_time")) if hydration.get("end_time") else (global_eh, global_em)

    # EYE
    eye_enabled = eye.get("enabled", False)
    eye_interval = (eye.get("interval_minutes") or 20) * 60 * 1000
    eye_duration = (eye.get("duration_seconds") or 20) * 1000

    # STRETCH
    stretch_enabled = stretch.get("enabled", False)
    stretch_interval = (stretch.get("interval_minutes") or 60) * 60 * 1000

    # WALK
    walk_enabled = walk.get("enabled", False)
    walk_interval = (walk.get("interval_minutes") or 60) * 60 * 1000

    # DND
    dnd = {
        "enabled": False,
        "sh": 0, "sm": 0,
        "eh": 0, "em": 0
    }

    if isinstance(dnd_list, list) and len(dnd_list) > 0:
        first = dnd_list[0]
        sh_d, sm_d = parse_time(first.get("start"))
        eh_d, em_d = parse_time(first.get("end"))

        dnd.update({
            "enabled": True,
            "sh": sh_d,
            "sm": sm_d,
            "eh": eh_d,
            "em": em_d
        })

    schedule = generate_schedule(tasks, (global_sh, global_sm), (global_eh, global_em), dnd)

    return {
        "hydration": {
            "enabled": h_enabled,
            "interval_ms": h_interval,
            "start_hour": sh,
            "start_min": sm,
            "end_hour": eh,
            "end_min": em
        },
        "eye": {
            "enabled": eye_enabled,
            "interval_ms": eye_interval,
            "duration_ms": eye_duration
        },
        "stretch": {
            "enabled": stretch_enabled,
            "interval_ms": stretch_interval
        },
        "walk": {
            "enabled": walk_enabled,
            "interval_ms": walk_interval
        },
        "dnd": dnd,
        "schedule": schedule
    }


# =========================
# ROUTE
# =========================
@app.route("/parse", methods=["POST"])
def parse_schedule():
    try:
        logs = []

        def log(step):
            logs.append(step)

        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Missing text"}), 400

        user_text = data.get("text")

        log("Calling OpenAI")

        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
        )

        raw_text = response.output_text
        log("Model responded")

        parsed = safe_json_parse(raw_text)
        parsed = ensure_defaults(parsed)

        log("Parsed safely")

        final_output = convert_to_device_schema(parsed)

        return jsonify({
            "data": final_output,
            "logs": logs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(port=5000)