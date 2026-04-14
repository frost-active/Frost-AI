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
            "abs": {
                "enabled": False,
                "times": []
            }
        },
        "stretch": {"enabled": False},
        "eye": {"enabled": False},
        "dnd": {"enabled": False},
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