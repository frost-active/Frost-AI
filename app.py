import json
import os
from datetime import datetime
from flask_cors import CORS
from flask import Flask, request, jsonify
from openai import OpenAI


app = Flask(__name__)
CORS(app)




# Secure API key from environment variable
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
- Follow this schema exactly


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
        user_text = request.json.get("text")


        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
        )


        raw_output = response.output_text.strip()
        parsed_json = json.loads(raw_output)


        return jsonify(parsed_json)


    except Exception as e:
        return jsonify({"error": str(e)}), 400




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)