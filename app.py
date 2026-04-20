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

Extract ONLY what the user is MODIFYING.

Rules:
- Return partial JSON (only what user mentions)
- Do NOT reset existing values
- Always return valid JSON

OUTPUT:
{
  "active_window": {},
  "tasks": [],
  "do_not_disturb": []
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
# RESILIENT JSON
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

    return {}


# =========================
# NORMALIZE TASKS
# =========================
def normalize_tasks(tasks):
    if not isinstance(tasks, list):
        return []

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
# 🧠 MERGE LOGIC (CORE)
# =========================
def merge_configs(old, new):

    if not old:
        return new

    merged = old.copy()

    # Merge tasks
    old_tasks = normalize_tasks(old.get("tasks", []))
    new_tasks = normalize_tasks(new.get("tasks", []))

    task_map = {t["type"]: t for t in old_tasks}

    for t in new_tasks:
        task_map[t["type"]] = t  # overwrite or add

    merged["tasks"] = list(task_map.values())

    # Merge active window
    if new.get("active_window"):
        merged["active_window"] = new["active_window"]

    # Merge DND
    if new.get("do_not_disturb"):
        merged["do_not_disturb"] = new["do_not_disturb"]

    return merged


# =========================
# SCHEDULER (same as before)
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

    filtered = [e for e in raw_events if not is_in_dnd(e["time_min"], dnd)]
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
                final.append({**event, "time_min": new_time})
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
# DEVICE SCHEMA
# =========================
def convert(parsed):

    tasks = normalize_tasks(parsed.get("tasks", []))
    active = parsed.get("active_window", {})

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

    dnd = {"enabled": False, "sh": 0, "sm": 0, "eh": 0, "em": 0}

    if parsed.get("do_not_disturb"):
        first = parsed["do_not_disturb"][0]
        sh, sm = parse_time(first.get("start"))
        eh, em = parse_time(first.get("end"))

        dnd.update({"enabled": True, "sh": sh, "sm": sm, "eh": eh, "em": em})

    schedule = generate_schedule(tasks, (global_sh, global_sm), (global_eh, global_em), dnd)

    return {
        "hydration": hydration,
        "eye": eye,
        "stretch": stretch,
        "walk": walk,
        "dnd": dnd,
        "schedule": schedule
    }


# =========================
# ROUTE
# =========================
@app.route("/parse", methods=["POST"])
def parse_schedule():
    try:
        data = request.get_json()

        user_text = data.get("text")
        previous = data.get("previous", {})

        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
        )

        raw_text = response.output_text

        new_parsed = safe_json_parse(raw_text)

        merged = merge_configs(previous, new_parsed)

        final_output = convert(merged)

        return jsonify({
            "data": final_output,
            "memory": merged
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(port=5000)