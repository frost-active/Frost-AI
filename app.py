import json
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CHAT_PROMPT = """
You are a helpful, friendly hydration and wellness assistant.
Hold natural conversations with the user.
Answer questions conversationally and clearly.
Do NOT return JSON unless explicitly asked.
"""

PARSE_PROMPT = """
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
    return "Hydration Chatbot API is running 🚀"

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        user_text = data.get("text")
        mode = data.get("mode", "chat")

        if not user_text:
            return jsonify({"error": "Missing 'text' field"}), 400

        if mode == "auto":
            keywords = ["remind", "schedule", "every", "from", "until", "between"]
            if any(word in user_text.lower() for word in keywords):
                mode = "parse"
            else:
                mode = "chat"

        system_prompt = PARSE_PROMPT if mode == "parse" else CHAT_PROMPT

        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ]
        )

        output_text = response.output_text.strip()

        if mode == "parse":
            try:
                parsed_json = json.loads(output_text)
                return jsonify({
                    "mode": "parse",
                    "data": parsed_json
                })
            except json.JSONDecodeError:
                return jsonify({
                    "error": "Invalid JSON returned by model",
                    "raw_output": output_text
                }), 500

        return jsonify({
            "mode": "chat",
            "message": output_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)