import os
import json
from datetime import datetime
from flask_cors import CORS
from flask import Flask, request, jsonify, Response
from openai import OpenAI

app = Flask(__name__)
app.json.sort_keys = False
CORS(app)

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    timeout=10,
    max_retries=2
)

SYSTEM_PROMPT = """
You are a scheduling assistant.

Extract hydration scheduling AND reminder timer information from user input.

Rules:
- Always return ONLY VALID JSON (no explanation)
- Use 24-hour time format (HH:MM)
- If a field is missing, use null
- If the user implies reminders, enable hydration_timer
- Default interval_minutes to 30 if not specified
- hydration_timer times should match active_window
- do_not_disturb only if explicitly mentioned
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
  "do_not_disturb": [],
  "exclusions": []
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

        # Input validation
        if not isinstance(user_text, str) or not user_text.strip():
            return jsonify({"error": "Invalid input text"}), 400

        if len(user_text) > 1000:
            return jsonify({"error": "Input too long"}), 400

        # OpenAI call
        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
        )

        # Extract text safely
        raw_output = response.output_text

        try:
            parsed = json.loads(raw_output)
        except Exception:
            return jsonify({
                "error": "Model did not return valid JSON",
                "raw_output": raw_output
            }), 500

        active_window = parsed.get("active_window", {})
        hydration_timer = parsed.get("hydration_timer", {})

        output = {
            "task": parsed.get("task", "hydration"),
            "active_window": {
                "start": active_window.get("start"),
                "end": active_window.get("end")
            },
            "hydration_timer": {
                "enabled": hydration_timer.get("enabled", False),
                "interval_minutes": hydration_timer.get("interval_minutes"),
                "start_time": hydration_timer.get("start_time"),
                "end_time": hydration_timer.get("end_time"),
                "alert_message": hydration_timer.get("alert_message", "Time to drink water 💧")
            },
            "do_not_disturb": parsed.get("do_not_disturb", []),
            "exclusions": parsed.get("exclusions", [])
        }

        print("Processing time:", datetime.now() - start_time)

        return Response(
            response=json.dumps(output, indent=2),
            mimetype="application/json",
            headers={
                "Content-Disposition": "attachment; filename=hydration_schedule.json"
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)