import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
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
- Always return VALID JSON
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

@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.route("/")
def home():
    return "Hydration Scheduler API is running 🚀"

@app.route("/parse", methods=["POST", "OPTIONS"])
def parse_schedule():
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response

    start_time = datetime.now()
    logs = []

    try:
        logs.append("Request received")

        data = request.get_json()
        logs.append("JSON parsed")

        user_text = data.get("text") if data else None
        logs.append(f"User input: {user_text}")

        if not isinstance(user_text, str) or not user_text.strip():
            logs.append("Invalid input")
            return jsonify({
                "error": "Invalid input text",
                "logs": logs
            }), 400

        logs.append("Calling OpenAI API")

        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
        )

        logs.append("Received response from OpenAI")

        raw_output = response.output_text.strip()

        if raw_output.startswith("```"):
            raw_output = raw_output.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw_output)
        logs.append("Parsed JSON")

        duration = (datetime.now() - start_time).total_seconds()
        logs.append(f"Processing time: {duration}s")

        return jsonify({
            "logs": logs,
            "data": parsed
        })

    except Exception as e:
        logs.append("Unhandled exception")
        logs.append(str(e))
        return jsonify({
            "error": "Internal server error",
            "logs": logs
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)