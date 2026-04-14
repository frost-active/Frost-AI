import os
import json
from datetime import datetime
from flask_cors import CORS
from flask import Flask, request, jsonify, Response
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

Extract hydration schedule, reminder intervals, do-not-disturb windows, and exclusions from user input.

Return ONLY valid JSON.


Rules:

- Always return VALID JSON only (no text, no explanation)
- Use 24-hour time format (HH:MM)
- If a field is missing, use null or empty list
- If reminders are implied, set hydration_timer.enabled = true
- Default interval_minutes = 30 if not specified


IMPORTANT EXTRACTION RULES:

- ALWAYS extract do_not_disturb if user says:
  "don't notify", "avoid", "no reminders", "mute", etc.

- ALWAYS extract exclusions if user mentions specific times to skip:
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


EXAMPLES:

Input:
"I work 9 to 5, remind me every 40 minutes"

Output:
{
  "task": "hydration",
  "parsable": true,
  "active_window": { "start": "09:00", "end": "17:00" },
  "hydration_timer": {
    "enabled": true,
    "interval_minutes": 40,
    "start_time": "09:00",
    "end_time": "17:00",
    "alert_message": "Time to drink water 💧"
  },
  "do_not_disturb": [],
  "exclusions": []
}



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

@app.route("/")
def home():
    return "Hydration Scheduler API is running 🚀"


@app.route("/parse", methods=["POST"])
def parse_schedule():
    try:
        start_time = datetime.now()

        # ✅ LOGGING SETUP
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

        # 🔥 OpenAI call
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

        # 🔥 Safe JSON parsing
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

        print("Processing time:", datetime.now() - start_time)

        # ✅ RETURN DATA + LOGS
        return jsonify({
            "data": output,
            "logs": logs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)