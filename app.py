import os
import json
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
# SYSTEM PROMPT (REFINED)
# =========================
SYSTEM_PROMPT = """
You are a highly precise scheduling extraction engine.

Return ONLY valid JSON.

Always return:
{
  "active_window": {},
  "tasks": [],
  "do_not_disturb": [],
  "exclusions": []
}

Rules:
- Use 24-hour format HH:MM
- Never omit a mentioned task
- Always include interval_minutes

Tasks:
- hydration → default 30 min
- eye → default 20 min, duration 20 sec
- stretch → default 60 min
- walk → default 90 min

Each task:
{
  "type": "hydration | eye | stretch | walk",
  "enabled": true,
  "interval_minutes": number,
  "duration_seconds": number or null,
  "start_time": "HH:MM" or null,
  "end_time": "HH:MM" or null
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


def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        return {
            "active_window": {},
            "tasks": [],
            "do_not_disturb": [],
            "exclusions": []
        }


# =========================
# NORMALIZE TASKS
# =========================
def normalize_tasks(parsed):
    tasks = parsed.get("tasks")

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
# ENGINE
# =========================
PRIORITY = {
    "hydration": 1,
    "eye": 2,
    "stretch": 3,
    "walk": 3
}

MIN_GAP = 5  # minutes


def is_in_dnd(time_min, dnd):
    if not dnd["enabled"]:
        return False

    start = dnd["sh"] * 60 + dnd["sm"]
    end = dnd["eh"] * 60 + dnd["em"]

    # FIX: handle overnight DND
    if start <= end:
        return start <= time_min <= end
    else:
        return time_min >= start or time_min <= end


def generate_schedule(tasks, global_start, global_end, dnd):
    raw_events = []

    global_start_min = global_start[0] * 60 + global_start[1]
    global_end_min = global_end[0] * 60 + global_end[1]

    # Step 1: Generate events (respect per-task windows)
    for t in tasks:
        if not t.get("enabled"):
            continue

        interval = t.get("interval_minutes")

        # FIX: do not skip valid 0-like values incorrectly
        if interval is None:
            continue

        # FIX: respect per-task start/end
        sh, sm = parse_time(t.get("start_time")) if t.get("start_time") else global_start
        eh, em = parse_time(t.get("end_time")) if t.get("end_time") else global_end

        start_min = sh * 60 + sm
        end_min = eh * 60 + em

        current = start_min + interval

        while current <= end_min:
            raw_events.append({
                "time_min": current,
                "task": t.get("type"),
                "priority": PRIORITY.get(t.get("type"), 99)
            })
            current += interval

    # Step 2: Remove DND events
    filtered = [
        e for e in raw_events
        if not is_in_dnd(e["time_min"], dnd)
    ]

    # Step 3: Sort
    filtered.sort(key=lambda x: (x["time_min"], x["priority"]))

    final = []

    for event in filtered:
        if not final:
            final.append(event)
            continue

        last = final[-1]

        # FIX: do NOT shift time → just skip if too close
        if event["time_min"] - last["time_min"] < MIN_GAP:
            continue
        else:
            final.append(event)

    # Step 4: Format
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
def convert_to_device_schema(parsed):

    tasks = normalize_tasks(parsed)

    active = parsed.get("active_window") or {}
    dnd_list = parsed.get("do_not_disturb") or []

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

    # FINAL SCHEDULE
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
# ROUTES
# =========================
@app.route("/")
def home():
    return "Adaptive Scheduler Engine 🚀"


@app.route("/parse", methods=["POST"])
def parse_schedule():
    try:
        start_time = datetime.now(pytz.utc).astimezone(IST)
        logs = []

        def log(step):
            current_time = datetime.now(pytz.utc).astimezone(IST).strftime('%H:%M:%S')
            logs.append(f"{current_time} - {step}")

        log("Request received")

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
        log("Model response received")

        parsed = safe_json_parse(raw_text)
        log("JSON parsed safely")

        final_output = convert_to_device_schema(parsed)

        log("Converted to schema")

        processing_time = datetime.now(pytz.utc).astimezone(IST) - start_time
        log(f"Total time: {round(processing_time.total_seconds(), 2)}s")

        return jsonify({"data": final_output, "logs": logs})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)