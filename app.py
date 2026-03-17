import json
import os
from datetime import datetime
from flask_cors import CORS
from flask import Flask, request, jsonify, Response
from openai import OpenAI
from collections import OrderedDict

app = Flask(__name__)
app.json.sort_keys = False
CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
You are a scheduling assistant.

Extract hydration scheduling AND reminder timer information from user input.

Rules:
- Always return VALID JSON
- Use 24-hour time format (HH:MM)
- If a field is missing, use null
- If the user implies reminders (e.g., "remind me", "every 30 minutes"), enable hydration_timer
- If no interval is mentioned but reminders are implied, default interval_minutes to 30
- hydration_timer.start_time and end_time should usually match active_window
- "do_not_disturb" represents time ranges where hydration reminders should pause
- dont include lunch breaks as do_not_disturb unless explicitly mentioned by the user
- dont consider sleep time as do_not_disturb unless explicitly mentioned by the user
- donot include meetings as do_not_disturb unless explicitly mentioned by the user
- Multiple do_not_disturb windows are allowed

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
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
        )

        raw_output = response.output_text.strip()

        if raw_output.startswith("```"):
            raw_output = raw_output.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw_output)

        active_window = parsed.get("active_window", {})
        hydration_timer = parsed.get("hydration_timer", {})

        ordered_output = OrderedDict()

        ordered_output["task"] = parsed.get("task", "hydration")

        ordered_output["active_window"] = {
            "start": active_window.get("start"),
            "end": active_window.get("end")
        }

        ordered_output["hydration_timer"] = {
            "enabled": hydration_timer.get("enabled", False),
            "interval_minutes": hydration_timer.get("interval_minutes"),
            "start_time": hydration_timer.get("start_time"),
            "end_time": hydration_timer.get("end_time"),
            "alert_message": hydration_timer.get("alert_message", "Time to drink water 💧")
        }

        ordered_output["do_not_disturb"] = parsed.get("do_not_disturb", [])
        ordered_output["exclusions"] = parsed.get("exclusions", [])

        print("Processing time:", datetime.now() - start_time)

        json_data = json.dumps(ordered_output, indent=2)

        return Response(
            json_data,
            mimetype="application/json",
            headers={
                "Content-Disposition": "attachment; filename=hydration_schedule.json"
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)