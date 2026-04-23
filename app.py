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
# SYSTEM PROMPT (UPDATED WITH DND)
# =========================
SYSTEM_PROMPT = """
You are a scheduling assistant.

Return ONLY valid JSON.

SUPPORTED TASK TYPES:
hydration, eye, stretch, walk

ALSO EXTRACT DO NOT DISTURB (DND):
Examples:
"no interruptions from 1 to 3"
"dnd 14:00 to 16:00"
"avoid reminders between 9 and 11"

FORMAT:
{
  "active_window": {
    "start": "HH:MM",
    "end": "HH:MM"
  },
  "tasks": [],
  "do_not_disturb": [
    {
      "start": "HH:MM",
      "end": "HH:MM"
    }
  ],
  "exclusions": []
}
"""


# =========================
# BASE COMPANY CONFIG
# =========================
BASE_CONFIG = {
  "tone_mode": "professional",
  "ui": {
    "action_log": {"enabled": True, "show_ms": 3000}
  },
  "dfplayer": {
    "volume": 24,
    "boot_volume": 15,
    "night_volume": 8,
    "night_start_hour": 22,
    "night_end_hour": 7,
    "night_mode_enabled": False
  },
  "audio": {
    "pomo_focus_music_enabled": True,
    "pomo_focus_music_track": 101,
    "pomo_focus_music_loop": True,
    "meditation_music_enabled": True,
    "meditation_music_track": 31
  },
  "clean": {
    "enabled": True,
    "soft_after_days": 2,
    "hard_after_days": 3,
    "soft_repeat_min": 60,
    "sticky_hard": True,
    "allow_device_ack": True,
    "trigger_hour": 17,
    "trigger_min": 0
  },
  "pomo": {
    "enabled": False,
    "focus_min": 25,
    "break_min": 5,
    "cycles": 4,
    "lap_mode_enabled": True,
    "laps": []
  },
  "healing": {"enabled": False, "require_dock": False, "play_min": 25, "default_track": 18, "slots": []},
  "meditation": {"enabled": False, "sh": 0, "sm": 0, "eh": 0, "em": 0, "display_sec": 600, "days": []},
  "medication_cfg": {"enabled": False, "require_ack": True, "allow_device_ack": True, "snooze_min": 15, "default_window_min": 120, "show_ms": 60000},
  "medication": [],
  "custom": {"enabled": False, "require_ack": True, "snooze_min": 5, "events": []},
  "ack_config": {"force_mode": False},
  "custom_texts": {
    "hydration": "Time to drink water!",
    "stretch": "Time to stretch!",
    "eye": "Time for eye break!",
    "walk": "Time for a short walk!"
  },
  "images": {},
  "priority": []
}


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
            return None
        h, m = t.split(":")
        return int(h), int(m)
    except:
        return None


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
            "start_time": parse_time(t.get("start_time")),
            "end_time": parse_time(t.get("end_time"))
        })

    return normalized


# =========================
# FINAL CONVERSION
# =========================
def convert_to_device_schema(parsed):

    tasks = normalize_tasks(parsed)
    active = parsed.get("active_window") or {}

    global_start = parse_time(active.get("start"))
    global_end = parse_time(active.get("end"))

    config = json.loads(json.dumps(BASE_CONFIG))

    config["_meta"] = {
        "schema_ver": None,
        "device": "FROST",
        "ts_written": 0
    }

    # DND DEFAULT
    config["dnd"] = {
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

    # TASK DEFAULTS (UNCHANGED)
    config["hydration"] = {
        "enabled": False,
        "mode": "interval",
        "interval_ms": 1800000,
        "prompt_duration_ms": 60000,
        "prompt_gap_ms": 600000,
        "require_ack": True,
        "goal_ml": 2000,
        "start_hour": 7,
        "start_min": 0,
        "end_hour": 22,
        "end_min": 0,
        "days": [],
        "abs": {"enabled": False, "times": []}
    }

    config["eye"] = {
        "enabled": False,
        "mode": "interval",
        "interval_ms": 1800000,
        "require_ack": True,
        "start_hour": 8,
        "start_min": 0,
        "end_hour": 20,
        "end_min": 0,
        "days": [],
        "abs": {"enabled": False, "times": []}
    }

    config["stretch"] = {
        "enabled": False,
        "mode": "interval",
        "interval_ms": 3600000,
        "duration_ms": 60000,
        "require_ack": True,
        "days": [],
        "phases": [],
        "abs": {"enabled": False, "times": []}
    }

    config["walk"] = {
        "enabled": False,
        "mode": "interval",
        "interval_min": 120,
        "display_sec": 90,
        "require_ack": True,
        "start_hour": 8,
        "start_min": 0,
        "end_hour": 20,
        "end_min": 0,
        "days": [],
        "abs": {"enabled": False, "times": []}
    }

    # APPLY TASKS (UNCHANGED)
    for t in tasks:
        start = t["start_time"] or global_start
        end = t["end_time"] or global_end

        sh, sm = start if start else (0, 0)
        eh, em = end if end else (23, 59)

        if t["type"] == "hydration":
            config["hydration"].update({
                "enabled": True,
                "interval_ms": (t["interval_minutes"] or 30) * 60000,
                "start_hour": sh,
                "start_min": sm,
                "end_hour": eh,
                "end_min": em
            })

        elif t["type"] == "eye":
            config["eye"].update({
                "enabled": True,
                "interval_ms": (t["interval_minutes"] or 20) * 60000,
                "start_hour": sh,
                "start_min": sm,
                "end_hour": eh,
                "end_min": em
            })

        elif t["type"] == "stretch":
            config["stretch"].update({
                "enabled": True,
                "interval_ms": (t["interval_minutes"] or 60) * 60000,
                "phases": [{
                    "sh": sh,
                    "sm": sm,
                    "eh": eh,
                    "em": em
                }]
            })

        elif t["type"] == "walk":
            config["walk"].update({
                "enabled": True,
                "interval_min": t["interval_minutes"] or 120,
                "start_hour": sh,
                "start_min": sm,
                "end_hour": eh,
                "end_min": em
            })

    # APPLY DND
    dnd_list = parsed.get("do_not_disturb") or []

    if isinstance(dnd_list, list) and len(dnd_list) > 0:
        first = dnd_list[0]
        start = parse_time(first.get("start"))
        end = parse_time(first.get("end"))

        if start and end:
            config["dnd"].update({
                "enabled": True,
                "sh": start[0],
                "sm": start[1],
                "eh": end[0],
                "em": end[1]
            })

    return config


# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return "Adaptive Scheduler Engine 🚀"


@app.route("/parse", methods=["POST"])
def parse_schedule():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Missing text"}), 400

        user_text = data.get("text")

        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
        )

        raw_text = response.output_text
        parsed = safe_json_parse(raw_text)
        final_output = convert_to_device_schema(parsed)

        return jsonify({"data": final_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)