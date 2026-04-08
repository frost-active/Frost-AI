import os
import json
import re
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
- ONLY RETURN JSON. NO EXTRA TEXT.

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
    start_time = datetime.now()
    logs = []

    try:
        logs.append("Request received")

        try:
            data = request.get_json(force=True)
        except:
            data = json.loads(request.data.decode("utf-8"))

        logs.append("JSON parsed")

        user_text = data.get("text") if data else None
        logs.append(f"User input: {user_text}")

        if not isinstance(user_text, str) or not user_text.strip():
            return jsonify({"error": "Invalid input", "logs": logs}), 400

        logs.append("Calling OpenAI API")

        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
        )

        logs.append("Received response")

        raw_output = response.output_text.strip().replace("\n", "")

        try:
            parsed = json.loads(raw_output)
            logs.append("Parsed JSON directly")
        except:
            logs.append("Direct parse failed, attempting recovery")

            match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                logs.append("Recovered JSON")
            else:
                return jsonify({
                    "error": "Invalid JSON from AI",
                    "logs": logs,
                    "raw_output": raw_output
                }), 500

        duration = (datetime.now() - start_time).total_seconds()
        logs.append(f"Processing time: {duration}s")

        return jsonify({
            "logs": logs,
            "data": parsed
        })

    except Exception as e:
        logs.append(str(e))
        return jsonify({"error": "Internal error", "logs": logs}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)