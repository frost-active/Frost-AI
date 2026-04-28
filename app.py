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
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are a strict scheduling assistant.

Return ONLY valid JSON. No explanation text.

SUPPORTED TASK TYPES:
hydration, eye, stretch, walk

MEDICATION FORMAT:
{
  "label": "string",
  "start": "YYYY-MM-DD",
  "end": "YYYY-MM-DD",
  "days": ["mon","tue","wed","thu","fri","sat","sun"],
  "times": ["HH:MM","HH:MM"]
}

RULES:
- Always use 24-hour HH:MM
- Normalize days to short form (mon, tue, ...)
- If days missing → assume all days

DATE HANDLING:
- If user says "for X days":
    start = today
    end = today + X days
- If user specifies start date:
    use that
- If both start and end provided:
    use both
- If nothing provided:
    return null for start and end

FINAL FORMAT:
{
  "active_window": {"start": "HH:MM", "end": "HH:MM"},
  "tasks": [],
  "medication": [],
  "do_not_disturb": [],
  "exclusions": []
}
"""


# =========================
# BASE CONFIG (FULL, SAFE)
# =========================
BASE_CONFIG = {
  "_meta": {"schema_ver": None, "device": "FROST", "ts_written": 0},
  "tone_mode": "professional",
  "ui": {"action_log": {"enabled": True, "show_ms": 3000}},
  "dfplayer": {
    "volume": 24, "boot_volume": 15, "night_volume": 8,
    "night_start_hour": 22, "night_end_hour": 7,
    "night_mode_enabled": False
  },
  "audio": {
    "pomo_focus_music_enabled": True,
    "pomo_focus_music_track": 101,
    "pomo_focus_music_loop": True,
    "meditation_music_enabled": True,
    "meditation_music_track": 31
  },
  "hydration": {
    "enabled": False, "mode": "interval",
    "interval_ms": 7200000,
    "prompt_duration_ms": 60000,
    "prompt_gap_ms": 600000,
    "require_ack": True,
    "goal_ml": 2000,
    "start_hour": 7, "start_min": 0,
    "end_hour": 22, "end_min": 0,
    "days": [],
    "abs": {"enabled": False, "times": []}
  },
  "eye": {
    "enabled": False, "mode": "interval",
    "interval_ms": 1800000,
    "require_ack": True,
    "start_hour": 8, "start_min": 0,
    "end_hour": 20, "end_min": 0,
    "days": [],
    "abs": {"enabled": False, "times": []}
  },
  "stretch": {
    "enabled": False, "mode": "interval",
    "interval_ms": 3600000,
    "duration_ms": 60000,
    "require_ack": True,
    "days": [],
    "phases": [],
    "abs": {"enabled": False, "times": []}
  },
  "walk": {
    "enabled": False, "mode": "interval",
    "interval_min": 120,
    "display_sec": 90,
    "require_ack": True,
    "start_hour": 8, "start_min": 0,
    "end_hour": 20, "end_min": 0,
    "days": [],
    "abs": {"enabled": False, "times": []}
  },
  "dnd": {
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
  },
  "medication_cfg": {
    "enabled": False,
    "require_ack": True,
    "allow_device_ack": True,
    "snooze_min": 15,
    "default_window_min": 120,
    "show_ms": 60000
  },
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
def safe_int(val):
    try:
        return int(val)
    except:
        return None


def parse_time(t):
    try:
        h, m = t.split(":")
        return int(h), int(m)
    except:
        return None


def normalize_days(days):
    mapping = {
        "monday":"mon","mon":"mon",
        "tuesday":"tue","tue":"tue",
        "wednesday":"wed","wed":"wed",
        "thursday":"thu","thu":"thu",
        "friday":"fri","fri":"fri",
        "saturday":"sat","sat":"sat",
        "sunday":"sun","sun":"sun"
    }

    if not isinstance(days, list):
        return ["mon","tue","wed","thu","fri","sat","sun"]

    out = []
    for d in days:
        d = str(d).lower()
        if d in mapping:
            out.append(mapping[d])

    return out if out else ["mon","tue","wed","thu","fri","sat","sun"]


def resolve_dates(start, end):
    today = date.today()

    if start and end:
        return start, end
    if start and not end:
        return start, (today + timedelta(days=30)).isoformat()
    if not start and not end:
        return today.isoformat(), (today + timedelta(days=7)).isoformat()

    return start, end


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
# NORMALIZATION
# =========================
def normalize_tasks(parsed):
    tasks = parsed.get("tasks") or []
    out = []

    for t in tasks:
        out.append({
            "type": t.get("type"),
            "interval_minutes": safe_int(t.get("interval_minutes")),
            "start_time": parse_time(t.get("start_time")),
            "end_time": parse_time(t.get("end_time"))
        })

    return out


def normalize_medication(parsed):
    meds = parsed.get("medication") or []
    out = []

    for m in meds:
        doses = []
        for t in (m.get("times") or []):
            pt = parse_time(t)
            if pt:
                doses.append({"h": pt[0], "m": pt[1]})

        if not doses:
            continue

        start, end = resolve_dates(m.get("start"), m.get("end"))

        out.append({
            "label": m.get("label", "Medication"),
            "start": start,
            "end": end,
            "days": normalize_days(m.get("days")),
            "doses": doses
        })

    return out


def build_plan(parsed):
    return {
        "tasks": normalize_tasks(parsed),
        "medication": normalize_medication(parsed),
        "dnd": parsed.get("do_not_disturb", []),
        "active_window": parsed.get("active_window", {})
    }


# =========================
# CONVERTER
# =========================
def convert_to_device_schema(plan):

    config = json.loads(json.dumps(BASE_CONFIG))
    config["_meta"]["ts_written"] = int(datetime.now(IST).timestamp())

    active = plan.get("active_window") or {}
    global_start = parse_time(active.get("start"))
    global_end = parse_time(active.get("end"))

    for t in plan.get("tasks", []):
        start = t["start_time"] or global_start
        end = t["end_time"] or global_end

        sh, sm = start if start else (0, 0)
        eh, em = end if end else (23, 59)

        if t["type"] == "hydration":
            config["hydration"].update({
                "enabled": True,
                "interval_ms": (t["interval_minutes"] or 30) * 60000,
                "start_hour": sh, "start_min": sm,
                "end_hour": eh, "end_min": em
            })

        elif t["type"] == "eye":
            config["eye"].update({
                "enabled": True,
                "interval_ms": (t["interval_minutes"] or 20) * 60000,
                "start_hour": sh, "start_min": sm,
                "end_hour": eh, "end_min": em
            })

        elif t["type"] == "stretch":
            config["stretch"].update({
                "enabled": True,
                "interval_ms": (t["interval_minutes"] or 60) * 60000,
                "phases": [{"sh": sh, "sm": sm, "eh": eh, "em": em}]
            })

        elif t["type"] == "walk":
            config["walk"].update({
                "enabled": True,
                "interval_min": t["interval_minutes"] or 120,
                "start_hour": sh, "start_min": sm,
                "end_hour": eh, "end_min": em
            })

    if plan.get("medication"):
        config["medication_cfg"]["enabled"] = True
        config["medication"] = plan["medication"]

    return config


# =========================
# ROUTE
# =========================
@app.route("/parse", methods=["POST"])
def parse_schedule():
    logs = []

    try:
        data = request.get_json()
        logs.append("Received input")

        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": data["text"]}
            ]
        )

        raw = response.output_text
        logs.append(f"LLM output: {raw[:200]}")

        parsed = safe_json_parse(raw)
        plan = build_plan(parsed)
        config = convert_to_device_schema(plan)

        return jsonify({"data": config, "logs": logs})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/")
def home():
    return "Adaptive Scheduler Engine 🚀"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)