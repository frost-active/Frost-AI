import json
import os
from datetime import datetime
from flask_cors import CORS
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a scheduling assistant.

Extract hydration scheduling AND reminder timer information from user input.

Rules:
- Always return VALID JSON
- Use 24-hour time format (HH:MM)
- If a field is missing, use null
- If the user implies reminders enable hydration_timer
- If no interval is mentioned but reminders are implied default interval_minutes to 30
- hydration_timer.start_time and end_time should match active_window
- do_not_disturb represents time ranges where reminders pause
- lunch, sleep, or meetings should NOT be included unless explicitly stated
- Multiple do_not_disturb windows allowed

Schema:
{
  "task": "hydration",
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
  "do_not_disturb": [
    {
      "label": "string",
      "start": "HH:MM",
      "end": "HH:MM"
    }
  ],
  "exclusions": [
    {
      "label": "string",
      "start": "HH:MM",
      "end": "HH:MM"
    },
    {
      "label": "string",
      "time": "HH:MM"
    }
  ]
}
"""

@app.route("/")
def home():
    return "Hydration Scheduler API is running 🚀"

@app.route("/parse", methods=["POST"])
def parse_schedule():
    try:

        start_time = datetime.now()

        data = request.get_json()
        user_text = data.get("text") if data else None

        if not user_text:
            return jsonify({"error": "No input text provided"}), 400

        response = client.responses.create(
            model="gpt-5-nano",
            response_format={"type": "json_object"},
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
        )

        raw_output = response.output_text.strip()
        parsed = json.loads(raw_output)

        ordered_output = {
            "task": parsed.get("task", "hydration"),
            "active_window": parsed.get("active_window", {"start": None, "end": None}),
            "hydration_timer": parsed.get("hydration_timer", {}),
            "do_not_disturb": parsed.get("do_not_disturb", []),
            "exclusions": parsed.get("exclusions", [])
        }

        print("Processing time:", datetime.now() - start_time)

        return jsonify(ordered_output)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)