import os
import json
from datetime import datetime, date, timedelta
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
# SYSTEM PROMPT (IMPROVED + MEDICATION)
# =========================
SYSTEM_PROMPT = """
You are a strict scheduling assistant.

Return ONLY valid JSON. No explanation text.

SUPPORTED TASK TYPES:
hydration, eye, stretch, walk

MEDICATION SUPPORT:
Extract medication schedules separately.

MEDICATION FORMAT:
{
  "label": "string",
  "start": "YYYY-MM-DD",
  "end": "YYYY-MM-DD",
  "days": ["mon","tue","wed","thu","fri","sat","sun"],
  "times": ["HH:MM","HH:MM"]
}

RULES:
- Always normalize time to 24-hour HH:MM
- If days not mentioned → assume all days
- If start/end dates not mentioned → leave null
- Extract multiple medications if present

DO NOT DISTURB (DND):
Extract all DND ranges.

FINAL FORMAT:
{
  "active_window": {
    "start": "HH:MM",
    "end": "HH:MM"
  },
  "tasks": [],
  "medication": [],
  "do_not_disturb": [],
  "exclusions": []
}
"""


# =========================
# BASE CONFIG
# =========================
BASE_CONFIG = {
  "tone_mode": "professional",
  "ui": {"action_log": {"enabled": True, "show_ms": 3000}},
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
  "pomo": {"enabled": False, "focus_min": 25, "break_min": 5, "cycles": 4, "lap_mode_enabled": True, "laps": []},
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
    "walk": "Time for a short walk!",
    "medication": "Medication reminder"
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
            "medication": [],
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
            "start_time": parse_time(t.get("start_time")),
            "end_time": parse_time(t.get("end_time"))
        })

    return normalized


# =========================
# NORMALIZE MEDICATION
# =========================
def normalize_medication(parsed):
    meds = parsed.get("medication")
    if not isinstance(meds, list):
        return []

    today = date.today().isoformat()
    future = (date.today() + timedelta(days=30)).isoformat()

    normalized = []

    for m in meds:
        if not isinstance(m, dict):
            continue

        times = m.get("times") or []
        doses = []

        for t in times:
            parsed_time = parse_time(t)
            if parsed_time:
                doses.append({
                    "h": parsed_time[0],
                    "m": parsed_time[1]
                })

        if not doses:
            continue

        normalized.append({
            "label": m.get("label", "Medication"),
            "start": m.get("start") or today,
            "end": m.get("end") or future,
            "days": m.get("days", ["mon","tue","wed","thu","fri","sat","sun"]),
            "doses": doses
        })

    return normalized


# =========================
# CONVERT TO DEVICE SCHEMA
# =========================
def convert_to_device_schema(parsed):

    tasks = normalize_tasks(parsed)
    medications = normalize_medication(parsed)

    active = parsed.get("active_window") or {}
    global_start = parse_time(active.get("start"))
    global_end = parse_time(active.get("end"))

    config = json.loads(json.dumps(BASE_CONFIG))

    config["_meta"] = {
        "schema_ver": None,
        "device": "FROST",
        "ts_written": int(datetime.now(IST).timestamp())
    }

    # DND
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

    # APPLY TASKS
    for t in tasks:
        start = t["start_time"] or global_start
        end = t["end_time"] or global_end

        sh, sm = start if start else (0, 0)
        eh, em = end if end else (23, 59)

        if t["type"] == "hydration":
            config["hydration"] = {
                "enabled": True,
                "mode": "interval",
                "interval_ms": (t["interval_minutes"] or 30) * 60000,
                "start_hour": sh,
                "start_min": sm,
                "end_hour": eh,
                "end_min": em
            }

        elif t["type"] == "eye":
            config["eye"] = {
                "enabled": True,
                "mode": "interval",
                "interval_ms": (t["interval_minutes"] or 20) * 60000,
                "start_hour": sh,
                "start_min": sm,
                "end_hour": eh,
                "end_min": em
            }

        elif t["type"] == "stretch":
            config["stretch"] = {
                "enabled": True,
                "mode": "interval",
                "interval_ms": (t["interval_minutes"] or 60) * 60000,
                "phases": [{"sh": sh, "sm": sm, "eh": eh, "em": em}]
            }

        elif t["type"] == "walk":
            config["walk"] = {
                "enabled": True,
                "mode": "interval",
                "interval_min": t["interval_minutes"] or 120,
                "start_hour": sh,
                "start_min": sm,
                "end_hour": eh,
                "end_min": em
            }

    # APPLY MEDICATION
    if medications:
        config["medication_cfg"]["enabled"] = True
        config["medication"] = medications

    # APPLY DND (first window for now)
    dnd_list = parsed.get("do_not_disturb") or []
    if dnd_list:
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