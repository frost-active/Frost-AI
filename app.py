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

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

IST = pytz.timezone('Asia/Kolkata')

# =========================
# SYSTEM PROMPT
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
- NEVER add tasks not explicitly mentioned
- Defaults apply ONLY if the task is mentioned

Only supported tasks:
- hydration
- eye
- stretch
- walk

Task rules:
- hydration → default 30 min
- eye → default 20 min, duration 20 sec
- stretch → default 60 min
- walk → default 90 min

Each task:
{
  "type": "hydration | eye | stretch | walk",
  "enabled": true,
  "interval_minutes": number,
  "duration_seconds": number or None,
  "start_time": "HH:MM" or None,
  "end_time": "HH:MM" or None
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

        t = t.strip().lower()

        match = re.match(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', t)
        if match:
            h = int(match.group(1))
            m = int(match.group(2)) if match.group(2) else 0
            period = match.group(3)

            if period == "pm" and h != 12:
                h += 12
            if period == "am" and h == 12:
                h = 0

            return h, m

        h, m = t.split(":")
        return int(h), int(m)

    except:
        return 0, 0


def extract_dnd_from_text(text):
    match = re.search(
        r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(to|-)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?',
        text.lower()
    )

    if not match:
        return None

    def convert(h, m, period):
        h = int(h)
        m = int(m) if m else 0

        if period == "pm" and h != 12:
            h += 12
        if period == "am" and h == 12:
            h = 0

        return h, m

    sh = convert(match.group(1), match.group(2), match.group(3))
    eh = convert(match.group(5), match.group(6), match.group(7))

    return {
        "sh": sh[0],
        "sm": sh[1],
        "eh": eh[0],
        "em": eh[1]
    }


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
            "enabled": t.get("enabled") is True,
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

MIN_GAP = 5


def is_in_dnd(time_min, dnd):
    if not dnd["enabled"]:
        return False

    start = dnd["sh"] * 60 + dnd["sm"]
    end = dnd["eh"] * 60 + dnd["em"]

    if start <= end:
        return start <= time_min <= end
    else:
        return time_min >= start or time_min <= end


def generate_schedule(tasks, global_start, global_end, dnd):
    raw_events = []

    for t in tasks:
        if not t.get("enabled"):
            continue

        interval = t.get("interval_minutes")
        if interval is None:
            continue

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
            continue
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
def convert_to_device_schema(parsed, user_text):

    tasks = normalize_tasks(parsed)

    VALID_TASKS = {"hydration", "eye", "stretch", "walk"}
    tasks = [
        t for t in tasks
        if t.get("type") in VALID_TASKS and t.get("enabled")
    ]

    active = parsed.get("active_window") or {}
    dnd_list = parsed.get("do_not_disturb") or []

    global_sh, global_sm = parse_time(active.get("start"))
    global_eh, global_em = parse_time(active.get("end"))

    if (global_sh, global_sm) == (0, 0) and (global_eh, global_em) == (0, 0):
        if tasks:
            global_sh, global_sm = 9, 0
            global_eh, global_em = 17, 0

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

    h_enabled = hydration.get("enabled", False)
    h_interval = (hydration.get("interval_minutes") or 30) * 60 * 1000

    sh, sm = parse_time(hydration.get("start_time")) if hydration.get("start_time") else (global_sh, global_sm)
    eh, em = parse_time(hydration.get("end_time")) if hydration.get("end_time") else (global_eh, global_em)

    eye_enabled = eye.get("enabled", False)
    eye_interval = (eye.get("interval_minutes") or 20) * 60 * 1000
    eye_duration = (eye.get("duration_seconds") or 20) * 1000

    stretch_enabled = stretch.get("enabled", False)
    stretch_interval = (stretch.get("interval_minutes") or 60) * 60 * 1000

    walk_enabled = walk.get("enabled", False)
    walk_interval = (walk.get("interval_minutes") or 60) * 60 * 1000

    # ✅ FIXED DND BLOCK
    dnd = {
        "enabled": False,
        "sh": 0, "sm": 0,
        "eh": 0, "em": 0
    }

    dnd_applied = False

    if isinstance(dnd_list, list) and len(dnd_list) > 0:
        first = dnd_list[0]

        sh_d, sm_d = parse_time(first.get("start"))
        eh_d, em_d = parse_time(first.get("end"))

        if not ((sh_d, sm_d) == (0, 0) and (eh_d, em_d) == (0, 0)):
            dnd.update({
                "enabled": True,
                "sh": sh_d,
                "sm": sm_d,
                "eh": eh_d,
                "em": eh_d
            })
            dnd_applied = True

    if not dnd_applied:
        fallback = extract_dnd_from_text(user_text)
        if fallback:
            dnd.update({
                "enabled": True,
                "sh": fallback["sh"],
                "sm": fallback["sm"],
                "eh": fallback["eh"],
                "em": fallback["em"]
            })

    schedule = generate_schedule(tasks, (global_sh, global_sm), (global_eh, global_em), dnd)

    return {
        "_meta": {
    "schema_ver": None,
    "device": "FROST",
    "ts_written": 0
  },
  "tone_mode": "professional",
  "ui": {
    "action_log": {
      "enabled": true,
      "show_ms": 3000
    }
  },
  "dfplayer": {
    "volume": 24,
    "boot_volume": 15,
    "night_volume": 8,
    "night_start_hour": 22,
    "night_end_hour": 7,
    "night_mode_enabled": false
  },
  "audio": {
    "pomo_focus_music_enabled": true,
    "pomo_focus_music_track": 101,
    "pomo_focus_music_loop": true,
    "meditation_music_enabled": true,
    "meditation_music_track": 31
  },
  "hydration": {
    "enabled": false,
    "mode": "interval",
    "interval_ms": 7200000,
    "prompt_duration_ms": 60000,
    "prompt_gap_ms": 600000,
    "require_ack": true,
    "goal_ml": 2000,
    "start_hour": 7,
    "start_min": 0,
    "end_hour": 22,
    "end_min": 0,
    "days": [],
    "abs": {
      "enabled": false,
      "times": []
    }
  },
  "stretch": {
    "enabled": false,
    "mode": "interval",
    "interval_ms": 900000,
    "duration_ms": 60000,
    "require_ack": true,
    "days": [
      "mon",
      "tue",
      "wed",
      "thu",
      "fri"
    ],
    "phases": [
      {
        "sh": 9,
        "sm": 0,
        "eh": 17,
        "em": 0
      }
    ],
    "abs": {
      "enabled": false,
      "times": [
        {
          "h": 11,
          "m": 18
        }
      ]
    }
  },
  "eye": {
    "enabled": false,
    "mode": "interval",
    "interval_ms": 1800000,
    "require_ack": true,
    "start_hour": 8,
    "start_min": 0,
    "end_hour": 20,
    "end_min": 0,
    "days": [
      "mon",
      "tue",
      "wed",
      "thu",
      "fri"
    ],
    "abs": {
      "enabled": false,
      "times": [
        {
          "h": 11,
          "m": 20
        }
      ]
    }
  },
  "dnd": {
    "enabled": false,
    "sh": 23,
    "sm": 0,
    "eh": 6,
    "em": 0,
    "allow_med": true,
    "allow_hydration": false,
    "allow_stretch": false,
    "allow_eye": false,
    "allow_cleaning": false,
    "allow_walk": false,
    "allow_meditation": false,
    "allow_healing": false,
    "allow_custom": false,
    "allow_pomodoro": false
  },
  "clean": {
    "enabled": true,
    "soft_after_days": 2,
    "hard_after_days": 3,
    "soft_repeat_min": 60,
    "sticky_hard": true,
    "allow_device_ack": true,
    "trigger_hour": 17,
    "trigger_min": 0
  },
  "pomo": {
    "enabled": true,
    "focus_min": 25,
    "break_min": 5,
    "cycles": 4,
    "lap_mode_enabled": true,
    "laps": [
      {
        "sh": 9,
        "sm": 0,
        "eh": 12,
        "em": 0,
        "enabled": true
      }
    ]
  },
  "healing": {
    "enabled": false,
    "require_dock": false,
    "play_min": 25,
    "default_track": 18,
    "slots": []
  },
  "walk": {
    "enabled": false,
    "mode": "interval",
    "interval_min": 120,
    "display_sec": 90,
    "require_ack": true,
    "start_hour": 8,
    "start_min": 0,
    "end_hour": 20,
    "end_min": 0,
    "days": [
      "mon",
      "tue",
      "wed",
      "thu",
      "fri"
    ],
    "abs": {
      "enabled": false,
      "times": [
        {
          "h": 11,
          "m": 23
        }
      ]
    }
  },
  "meditation": {
    "enabled": true,
    "sh": 11,
    "sm": 45,
    "eh": 11,
    "em": 50,
    "display_sec": 600,
    "days": [
      "mon",
      "tue",
      "wed",
      "thu",
      "fri"
    ]
  },
  "medication_cfg": {
    "enabled": false,
    "require_ack": true,
    "allow_device_ack": true,
    "snooze_min": 15,
    "default_window_min": 120,
    "show_ms": 60000
  },
  "medication": [
    {
      "label": "Morining Vitamins",
      "start": "2026-04-16",
      "end": "2026-05-11",
      "days": [
        "mon",
        "tue",
        "wed",
        "thu",
        "fri",
        "sat",
        "sun"
      ],
      "doses": [
        {
          "h": 6,
          "m": 12
        },
        {
          "h": 11,
          "m": 30
        },
        {
          "h": 11,
          "m": 32
        }
      ]
    }
  ],
  "custom": {
    "enabled": false,
    "require_ack": true,
    "snooze_min": 5,
    "events": [
      {
        "h": 0,
        "m": 0,
        "label": "",
        "show_ms": 60000,
        "type": "absolute",
        "date": "2026-04-16"
      }
    ]
  },
  "ack_config": {
    "force_mode": false
  },
  "custom_texts": {
    "hydration": "Time to drink water!",
    "stretch": "Time to stretch!",
    "eye": "Time for eye break!",
    "walk": "Time for a short walk!",
    "medication": "Medication reminder",
    "healing": "Healing session",
    "custom": "Custom reminder",
    "clean_soft": "Bottle cleaning due (soft)",
    "clean_hard": "Bottle cleaning overdue (hard)",
    "meditation": "Meditation time",
    "pomodoro_focus": "Focus time started",
    "pomodoro_break": "Break time started"
  },
  "images": {
    "hydration_p1": "drinkwater1",
    "hydration_p2": "drinkwater2",
    "hydration_paused": "placeBottleImage_inverted",
    "eye": "rule",
    "stretch": "image_time_to_stretch_inverted",
    "medication": "image_medication_reminder_alt_inverted",
    "clean_soft": "cleen_bottel_image",
    "clean_hard": "clean_bottel_1",
    "meditation": "meditation",
    "healing": "waterRefillImage_inverted",
    "walk": "short_walk",
    "boot": "frost_logo",
    "clock_bg": "clock_bg",
    "clock_bg_dnd": "clock_bg_dnd",
    "pomodoro_focus_bg": "pomodoro_focus_bg",
    "pomodoro_break_bg": "pomodoro_break_bg",
    "custom_bg": "frost_logo",
    "hydration_data_screen": "wallpaper1",
    "bottle_missing": "placeBottleImage_inverted"
  },
  "priority": [
    "bottle_missing",
    "clean_hard",
    "clean_soft",
    "medication",
    "stretch",
    "eye",
    "hydration_paused",
    "hydration_p1",
    "hydration_p2",
    "hydration_data_screen",
    "meditation",
    "healing",
    "walk",
    "custom",
    "clock"
  ]
  }


# =========================
# ROUTES
# =========================
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

        final_output = convert_to_device_schema(parsed, user_text)

        log("Converted to schema")

        processing_time = datetime.now(pytz.utc).astimezone(IST) - start_time
        log(f"Total time: {round(processing_time.total_seconds(), 2)}s")

        return jsonify({"data": final_output, "logs": logs})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)