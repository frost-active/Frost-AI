import os
import json
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from flask_cors import CORS
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
app.json.sort_keys = False

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

IST = ZoneInfo("Asia/Kolkata")

# 🔒 Hardened system prompt
SYSTEM_PROMPT = """
You are a scheduling assistant.

STRICT RULES:
- ONLY output valid JSON
- NO explanations
- NO markdown
- NO trailing commas

Extract hydration schedule, reminder intervals, do-not-disturb windows, and exclusions.

Rules:
- Use 24-hour format HH:MM
- Default interval_minutes = 30
- If reminders implied → enabled = true
- Always extract DND if mentioned
- Always extract exclusions if specific times mentioned

OUTPUT FORMAT:

{
  "task": "hydration",
  "parsable": true,
  "active_window": { "start": "HH:MM", "end": "HH:MM" },
  "hydration_timer": {
    "enabled": true,
    "interval_minutes": number,
    "start_time": "HH:MM",
    "end_time": "HH:MM",
    "alert_message": "Time to drink water 💧"
  },
  "do_not_disturb": [],
  "exclusions": []
}
"""


# ✅ Normalization layer
def normalize(parsed):
    timer = parsed.get("hydration_timer", {}) or {}
    active = parsed.get("active_window", {}) or {}

    # interval safety
    interval = timer.get("interval_minutes") or 30
    timer["interval_minutes"] = int(interval)

    # active window fallback
    if not active.get("start") or not active.get("end"):
        active["start"] = "09:00"
        active["end"] = "18:00"

    # ensure alert message exists
    if not timer.get("alert_message"):
        timer["alert_message"] = "Time to drink water 💧"

    parsed["hydration_timer"] = timer
    parsed["active_window"] = active

    return parsed


def convert_to_device_schema(parsed):

    def parse_time(t):
        try:
            h, m = map(int, t.split(":"))
            return h, m
        except:
            return 0, 0

    active = parsed.get("active_window", {})
    timer = parsed.get("hydration_timer", {})
    dnd_list = parsed.get("do_not_disturb", [])
    exclusions = parsed.get("exclusions", [])

    sh, sm = parse_time(active.get("start"))
    eh, em = parse_time(active.get("end"))

    interval_ms = timer.get("interval_minutes", 30) * 60 * 1000

    dnd = {
        "enabled": False,
        "sh": 0, "sm": 0, "eh": 0, "em": 0,
        "allow_med": True,
        "allow_hydration": False
    }

    if dnd_list:
        first = dnd_list[0]
        sh_d, sm_d = parse_time(first.get("start"))
        eh_d, em_d = parse_time(first.get("end"))

        dnd.update({
            "enabled": True,
            "sh": sh_d, "sm": sm_d,
            "eh": eh_d, "em": em_d
        })

    abs_times = []
    for t in exclusions:
        h, m = parse_time(t)
        abs_times.append({"h": h, "m": m})

    return {
        "_meta": {
            "device": "FROST",
            "ts_written": int(datetime.now(IST).timestamp())
        },
        "custom_texts": {
            "hydration": timer.get("alert_message")
        },
        "hydration": {
            "enabled": bool(timer.get("enabled")),
            "interval_ms": interval_ms,
            "start_hour": sh,
            "start_min": sm,
            "end_hour": eh,
            "end_min": em,
            "abs": {
                "enabled": len(abs_times) > 0,
                "times": abs_times
            }
        },
        "dnd": dnd
    }


@app.route("/")
def home():
    return "Hydration Scheduler API is running 🚀"


@app.route("/parse", methods=["POST"])
def parse_schedule():
    try:
        start_time = datetime.now(IST)
        request_id = str(uuid.uuid4())[:8]
        logs = []

        def log(step, level="INFO"):
            current_time = datetime.now(IST).strftime('%H:%M:%S %Z')
            logs.append(f"[{level}] {current_time} [{request_id}] {step}")

        log("Request received")

        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"success": False, "error": "Missing 'text' field"}), 400

        user_text = data.get("text")

        if not isinstance(user_text, str) or not user_text.strip():
            return jsonify({"success": False, "error": "Invalid input"}), 400

        log("Calling OpenAI")

        # ✅ FIX: removed response_format
        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]
        )

        raw_text = response.output_text

        log("Received response from model")

        try:
            parsed = json.loads(raw_text)
            log("JSON parsed successfully")
        except Exception:
            log(f"INVALID JSON FROM MODEL: {raw_text}", "ERROR")
            return jsonify({
                "success": False,
                "error": "Model returned invalid JSON",
                "raw_output": raw_text
            }), 500

        parsed = normalize(parsed)
        log("Normalization complete")

        final_output = convert_to_device_schema(parsed)
        log("Schema conversion complete")

        total_time = round((datetime.now(IST) - start_time).total_seconds(), 2)

        return jsonify({
            "success": True,
            "config": final_output,
            "meta": {
                "processing_time_s": total_time,
                "request_id": request_id
            },
            "logs": logs
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)