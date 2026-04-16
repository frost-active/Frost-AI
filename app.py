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
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are a scheduling assistant.

Extract ALL scheduling tasks from user input.

Return ONLY valid JSON.

Supported tasks:
- hydration
- eye
- stretch
- walk

Output format:
{
  "tasks": [],
  "do_not_disturb": [],
  "exclusions": []
}

rules:
-timings with labels "lunch", "meetings", "break" are to be considered as do not disturb window
- if user mentions "do not disturb" or "dnd" or "focus mode" or similar, consider that as do not disturb window
"""


# =========================
# SAFE HELPERS
# =========================
def safe_int(val, default=0):
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

        t_type = t.get("type")

        normalized.append({
            "type": t_type,
            "enabled": bool(t.get("enabled", True)),
            "interval_minutes": safe_int(t.get("interval_minutes"), None),
            "duration_seconds": safe_int(t.get("duration_seconds"), None),
            "start_time": t.get("start_time"),
            "end_time": t.get("end_time")
        })

    return normalized


# =========================
# DEVICE SCHEMA CONVERSION
# =========================
def convert_to_device_schema(parsed):

    tasks = normalize_tasks(parsed)

    dnd_list = parsed.get("do_not_disturb") or []
    exclusions = parsed.get("exclusions") or []

    hydration = {}
    eye = {}
    stretch = {}
    walk = {}

    # -------------------------
    # TASK DISTRIBUTION
    # -------------------------
    for t in tasks:
        if t["type"] == "hydration":
            hydration = t

        elif t["type"] == "eye":
            eye = t

        elif t["type"] == "stretch":
            stretch = t

        elif t["type"] == "walk":
            walk = t

    # -------------------------
    # HYDRATION (SAFE DEFAULT)
    # -------------------------
    h_enabled = hydration.get("enabled", False)
    h_interval = (hydration.get("interval_minutes") or 30) * 60 * 1000

    sh, sm = parse_time(hydration.get("start_time"))
    eh, em = parse_time(hydration.get("end_time"))

    # -------------------------
    # EYE
    # -------------------------
    eye_enabled = eye.get("enabled", False)
    eye_interval = (eye.get("interval_minutes") or 20) * 60 * 1000
    eye_duration = (eye.get("duration_seconds") or 20) * 1000

    # -------------------------
    # STRETCH
    # -------------------------
    stretch_enabled = stretch.get("enabled", False)
    stretch_interval = (stretch.get("interval_minutes") or 60) * 60 * 1000

    # -------------------------
    # WALK
    # -------------------------
    walk_enabled = walk.get("enabled", False)
    walk_interval = (walk.get("interval_minutes") or 60) * 60 * 1000

    # -------------------------
    # DND
    # -------------------------
    dnd = {
        "enabled": False,
        "sh": 0, "sm": 0,
        "eh": 0, "em": 0,
        "allow_med": True,
        "allow_hydration": False,
        "allow_stretch": False,
        "allow_eye": False,
        "allow_cleaning": False,
        "allow_walk": False,
        "allow_meditation": False,
        "allow_healing": False,
        "allow_custom": False,
        "allow_pomodoro": False
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

    # -------------------------
    # EXCLUSIONS
    # -------------------------
    abs_times = []
    if isinstance(exclusions, list):
        for t in exclusions:
            h, m = parse_time(t)
            abs_times.append({"h": h, "m": m})

    abs_config = {
        "enabled": len(abs_times) > 0,
        "times": abs_times
    }

    # =========================
    # FINAL OUTPUT
    # =========================
    return {
        "_meta": {
            "schema_ver": None,
            "device": "FROST",
            "ts_written": 0
        },

        "hydration": {
            "enabled": h_enabled,
            "interval_ms": h_interval,
            "prompt_duration_ms": 60000,
            "prompt_gap_ms": 60000,
            "require_ack": True,
            "goal_ml": 2000,
            "start_hour": sh,
            "start_min": sm,
            "end_hour": eh,
            "end_min": em,
            "mode": "interval",
            "days": [],
            "abs": abs_config
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

        "clean": {"enabled": True},
        "pomo": {"enabled": False},
        "healing": {"enabled": False},
        "meditation": {"enabled": False},
        "custom": {"enabled": False},

        "priority": []
    }


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return "Stabilized Scheduler API 🚀"


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

        return jsonify({
            "data": final_output,
            "logs": logs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)