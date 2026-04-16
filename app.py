import os
import json
from datetime import datetime
import pytz
from flask_cors import CORS
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
app.json.sort_keys = False
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=True
)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

IST = pytz.timezone('Asia/Kolkata')

SYSTEM_PROMPT = """
You are a scheduling assistant.

Extract hydration schedule, reminder intervals, do-not-disturb windows, and exclusions from user input.

Return ONLY valid JSON.

Rules:

- Always return VALID JSON only
- Use 24-hour time format (HH:MM)
- If a field is missing, use null or empty list
- If reminders are implied, set hydration_timer.enabled = true
- Default interval_minutes = 30 if not specified

IMPORTANT EXTRACTION RULES:

- ALWAYS extract do_not_disturb if user says:
  "don't notify", "avoid", "no reminders", "mute", etc.

- ALWAYS extract exclusions if user mentions specific times:
  e.g. "skip 2:30 PM" → "14:30"

- Convert time ranges:
  "12 to 1 PM" → start: "12:00", end: "13:00"

- Convert single times:
  "2:30 PM" → "14:30"

- NEVER ignore time constraints

OUTPUT FORMAT:

{
  "task": "hydration",
  "parsable": true,
  "active_window": {
    "start": "HH:MM",
    "end": "HH:MM"
  },
  "hydration_timer": {
    "enabled": true,
    "interval_minutes": number,
    "start_time": "HH:MM",
    "end_time": "HH:MM",
    "alert_message": "Time to drink water 💧"
  },
  "do_not_disturb": [
    {
      "start": "HH:MM",
      "end": "HH:MM"
    }
  ],
  "exclusions": ["HH:MM"]
}

EXAMPLE:

Input:
"I sit from 8 to 4, remind me every 45 minutes, don't notify me from 12 to 1 PM, skip 2:30 PM"

Output:
{
  "task": "hydration",
  "parsable": true,
  "active_window": { "start": "08:00", "end": "16:00" },
  "hydration_timer": {
    "enabled": true,
    "interval_minutes": 45,
    "start_time": "08:00",
    "end_time": "16:00",
    "alert_message": "Time to drink water 💧"
  },
  "do_not_disturb": [
    { "start": "12:00", "end": "13:00" }
  ],
  "exclusions": ["14:30"]
}
"""


def convert_to_device_schema(parsed):

    def parse_time(t):
        if not t:
            return 0, 0
        h, m = t.split(":")
        return int(h), int(m)

    active = parsed.get("active_window", {})
    timer = parsed.get("hydration_timer", {})
    dnd_list = parsed.get("do_not_disturb", [])
    exclusions = parsed.get("exclusions", [])

    sh, sm = parse_time(active.get("start"))
    eh, em = parse_time(active.get("end"))

    interval_ms = timer.get("interval_minutes", 30) * 60 * 1000

    dnd = {
        "enabled": False,
        "sh": 0,
        "sm": 0,
        "eh": 0,
        "em": 0,
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

    if dnd_list:
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

    abs_times = []
    for t in exclusions:
        h, m = parse_time(t)
        abs_times.append({"h": h, "m": m})

    abs_config = {
        "enabled": len(abs_times) > 0,
        "times": abs_times
    }

    return {
        "_meta": {
            "schema_ver": None,
            "device": "FROST",
            "ts_written": 0
        },
        "dfplayer": {
            "volume": 30,
            "boot_volume": 15,
            "night_volume": 8,
            "night_start_hour": 22,
            "night_end_hour": 7,
            "night_mode_enabled": False
        },
        "audio": {
            "pomo_focus_music_enabled": True,
            "pomo_focus_music_track": 31,
            "pomo_focus_music_loop": True,
            "meditation_music_enabled": False,
            "meditation_music_track": 35
        },
        "tone_mode": "professional",
        "ui": {
            "action_log": {
                "enabled": True,
                "show_ms": 3000
            }
        },
        "custom_texts": {
            "hydration": "Time to drink water!"
        },
        "hydration": {
            "enabled": timer.get("enabled", False),
            "interval_ms": interval_ms,
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
        "dnd": dnd,
        "stretch": {"enabled": False},
        "eye": {"enabled": False},
        "clean": {"enabled": True},
        "pomo": {"enabled": False},
        "healing": {"enabled": False},
        "walk": {"enabled": False},
        "meditation": {"enabled": False},
        "medication": [],
        "medication_cfg": {"enabled": True},
        "custom": {"enabled": False},
        "images": {},
        "priority": []
    }


@app.route("/")
def home():
    return "Hydration Scheduler API is running 🚀"


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
            return jsonify({"error": "Missing 'text' field"}), 400

        user_text = data.get("text")

        if not isinstance(user_text, str) or not user_text.strip():
            return jsonify({"error": "Invalid input text"}), 400

        if len(user_text) > 1000:
            return jsonify({"error": "Input too long"}), 400

        log("Input validated")
        log("Calling OpenAI API")

        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
        )

        raw_text = response.output_text

        log("Received response from OpenAI")

        try:
            parsed = json.loads(raw_text)
            log("JSON parsed successfully")
        except Exception:
            return jsonify({
                "error": "Model returned invalid JSON",
                "raw_output": raw_text
            }), 500

        final_schema_output = convert_to_device_schema(parsed)

        log("Converted to device schema")
        log("Schema transformation complete")

        processing_time = datetime.now(pytz.utc).astimezone(IST) - start_time
        log(f"Total time: {round(processing_time.total_seconds(), 2)}s")

        return jsonify({
            "data": final_schema_output,
            "logs": logs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)