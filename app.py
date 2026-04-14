import os
import json
from datetime import datetime
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

SYSTEM_PROMPT = """
You are a scheduling assistant.

Extract hydration scheduling AND reminder timer information from user input.

Rules:
- Always return VALID JSON
- Use 24-hour time format (HH:MM)
- If a field is missing, use null
- If the user implies reminders, enable hydration_timer
- Default interval_minutes to 30 if not specified
- hydration_timer times should match active_window
- do_not_disturb only if explicitly mentioned
- Multiple do_not_disturb windows allowed
- flag invalid content as parsable=false
- ALWAYS extract do_not_disturb windows if user mentions "don't notify", "avoid", "no reminders"
- ALWAYS extract exclusions if user mentions specific times to skip
- NEVER ignore time constraints

Schema:
{
  "task": "hydration",
  "parsable": true,
  "active_window": {
    "start": "HH:MM",
    "end": "HH:MM"
  },
  "hydration_timer": {
    "enabled": true,
    "interval_minutes": 30,
    "start_time": "HH:MM",
    "end_time": "HH:MM",
    "alert_message": "Time to drink water 💧"
  },
  "do_not_disturb": [],
  "exclusions": []
}
"""

# ✅ TRANSFORMATION FUNCTION
def convert_to_device_schema(parsed):
    def parse_time(t):
        if not t:
            return 0, 0
        h, m = t.split(":")
        return int(h), int(m)

    active = parsed.get("active_window", {})
    timer = parsed.get("hydration_timer", {})

    sh, sm = parse_time(active.get("start"))
    eh, em = parse_time(active.get("end"))

    interval_ms = timer.get("interval_minutes", 30) * 60 * 1000

    return {
    {
  "_meta": {
    "schema_ver": null,
    "device": "FROST",
    "ts_written": 0
  },
  "dfplayer": {
    "volume": 30,
    "boot_volume": 15,
    "night_volume": 8,
    "night_start_hour": 22,
    "night_end_hour": 7,
    "night_mode_enabled": false
  },
  "audio": {
    "pomo_focus_music_enabled": true,
    "pomo_focus_music_track": 31,
    "pomo_focus_music_loop": true,
    "meditation_music_enabled": false,
    "meditation_music_track": 35
  },
  "tone_mode": "professional",
  "ui": {
    "action_log": {
      "enabled": true,
      "show_ms": 3000
    }
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
  "hydration": {
    "enabled": false,
    "interval_ms": 60000,
    "prompt_duration_ms": 60000,
    "prompt_gap_ms": 60000,
    "require_ack": true,
    "goal_ml": 2000,
    "start_hour": 0,
    "start_min": 0,
    "end_hour": 23,
    "end_min": 59,
    "mode": "absolute",
    "days": ["sat"],
    "abs": {
      "enabled": true,
      "times": [
        { "h": 12, "m": 50 }

      ]
    }
  },
  "stretch": {
    "enabled": false,
    "interval_ms": 60000,
    "duration_ms": 60000,
     "phases": [
      { "sh": 9, "sm": 0, "eh": 23, "em": 0 }
    ],
    "require_ack": false,
    "mode": "interval",
    "days": ["mon"],
    "abs": {
      "enabled": false,
      "times": [
        { "h": 13, "m": 17 }

      ]
    }

  },
  "eye": {
    "enabled": false,
    "interval_ms": 60000,
    "require_ack": true,
    "start_hour": 0,
    "start_min": 0,
    "end_hour": 23,
    "end_min": 59,
    "mode": "absolute",
    "days": [],
    "abs": {
      "enabled": true,
      "times": [
        { "h": 13, "m": 28 }

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
    "allow_device_ack": false,
    "trigger_hour": 17,
    "trigger_min": 0
  },
  "pomo": {
    "enabled": true,
    "focus_min": 25,
    "break_min": 5,
    "cycles": 1,
    "lap_mode_enabled": true,
    "laps": [
      {
        "sh": 14,
        "sm": 58,
        "eh": 15,
        "em": 30,
        "enabled": false
      }
      ]
  },
  "healing": {
    "enabled": false,
    "require_dock": true,
    "play_min": 25,
    "default_track": 18,
    "slots": [
      {
        "sh": 6,
        "sm": 30,
        "eh": 7,
        "em": 0,
        "track": 18,
        "repeat_min": 0,
        "days": []
      },
      {
        "sh": 20,
        "sm": 0,
        "eh": 21,
        "em": 0,
        "track": 18,
        "repeat_min": 0,
        "days": []
      }
    ]
  },
  "walk": {
    "enabled": true,
    "interval_min": 5,
    "display_sec": 90,
    "require_ack": true,
    "start_hour": 0,
    "start_min": 0,
    "end_hour": 23,
    "end_min": 59,
    "mode": "absolute",
    "days": [],
    "abs": {
      "enabled": true,
      "times": [
        { "h": 13, "m": 55 }

      ]
    }
  },
  "meditation": {
    "enabled": true,
    "sh": 20,
    "sm": 40,
    "eh": 7,
    "em": 0,
    "display_sec": 600,
    "days": [ "wed","thu"]
  },
  "medication": [
    {
      "label": "Sample Medication",
      "start": "2026-01-01",
      "end": "2026-12-31",
      "days": [],
      "doses": [
        { "h": 8, "m": 0 },
        { "h": 12, "m": 0 },
        { "h": 18, "m": 0 }
      ]
    }
  ],
  "medication_cfg": {
    "enabled": true,
    "require_ack": true,
    "allow_device_ack": false,
    "snooze_min": 15,
    "default_window_min": 120,
    "show_ms": 60000
  },
  "custom": {
    "enabled": true,
    "require_ack": true,
    "snooze_min": 5,
    "events": [
      {
        "label": "Water plants",
        "h": 9,
        "m": 0,
        "show_ms": 60000,
        "start": "2026-06-01",
        "end": "2026-09-30",
        "days": []
      }
    ]
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


@app.route("/")
def home():
    return "Hydration Scheduler API is running 🚀"


@app.route("/parse", methods=["POST"])
def parse_schedule():
    try:
        start_time = datetime.now()

        logs = []
        def log(step):
            logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {step}")

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

        print("RAW MODEL OUTPUT:", raw_text)

        try:
            parsed = json.loads(raw_text)
            log("JSON parsed successfully")
        except Exception:
            return jsonify({
                "error": "Model returned invalid JSON",
                "raw_output": raw_text
            }), 500

        active_window = parsed.get("active_window", {})
        hydration_timer = parsed.get("hydration_timer", {})

        output = {
            "task": parsed.get("task", "hydration"),
            "parsable": parsed.get("parsable", True),
            "active_window": {
                "start": active_window.get("start"),
                "end": active_window.get("end")
            },
            "hydration_timer": {
                "enabled": hydration_timer.get("enabled", False),
                "interval_minutes": hydration_timer.get("interval_minutes", 30),
                "start_time": hydration_timer.get("start_time"),
                "end_time": hydration_timer.get("end_time"),
                "alert_message": hydration_timer.get(
                    "alert_message",
                    "Time to drink water 💧"
                )
            },
            "do_not_disturb": parsed.get("do_not_disturb", []),
            "exclusions": parsed.get("exclusions", [])
        }

        log("Response ready")

        # ✅ TRANSFORM TO FINAL SCHEMA
        final_schema_output = convert_to_device_schema(output)
        log("Converted to device schema")
        log("Schema transformation complete")

        # ✅ TOTAL TIME LOG
        processing_time = datetime.now() - start_time
        log(f"Total time: {round(processing_time.total_seconds(), 2)}s")

        print("Processing time:", processing_time)

        return jsonify({
            "data": final_schema_output,
            "logs": logs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)