import os
import json
from datetime import datetime, date, timedelta
import pytz
from flask_cors import CORS
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
app.json.sort_keys = False

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

IST = pytz.timezone('Asia/Kolkata')


# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """
You are a strict scheduling assistant.

Return ONLY valid JSON. No explanation text.

SUPPORTED TASK TYPES:
hydration, eye, stretch, walk

MEDICATION FORMAT:
{
  "label": "string",
  "start": "YYYY-MM-DD",
  "end": "YYYY-MM-DD",
  "days": ["mon","tue","wed","thu","fri","sat","sun"],
  "times": ["HH:MM","HH:MM"]
}

RULES:
- Always use 24-hour HH:MM
- Normalize days to short form (mon, tue, ...)
- If days missing → assume all days

DATE HANDLING:
- If user says "for X days":
    start = today
    end = today + X days
- If user specifies start date:
    use that
- If both start and end provided:
    use both
- If nothing provided:
    return null for start and end

FINAL FORMAT:
{
  "active_window": {"start": "HH:MM", "end": "HH:MM"},
  "tasks": [],
  "medication": [],
  "do_not_disturb": [],
  "exclusions": []
}
"""


# =========================
# BASE CONFIG (UNCHANGED)
# =========================
BASE_CONFIG = { ... }  # KEEP YOUR FULL ORIGINAL CONFIG HERE


# =========================
# HELPERS
# =========================
def safe_int(val, default=None):
    try:
        return int(val)
    except:
        return default


def parse_time(t):
    try:
        if not t:
            return None
        h, m = t.split(":")
        return int(h), int(m)
    except:
        return None


def normalize_days(days):
    mapping = {
        "monday":"mon","mon":"mon",
        "tuesday":"tue","tue":"tue",
        "wednesday":"wed","wed":"wed",
        "thursday":"thu","thu":"thu",
        "friday":"fri","fri":"fri",
        "saturday":"sat","sat":"sat",
        "sunday":"sun","sun":"sun"
    }

    if not isinstance(days, list):
        return ["mon","tue","wed","thu","fri","sat","sun"]

    out = []
    for d in days:
        d = str(d).lower()
        if d in mapping:
            out.append(mapping[d])

    return out or ["mon","tue","wed","thu","fri","sat","sun"]


def resolve_dates(start, end):
    today = date.today()

    if start and end:
        return start, end

    if start and not end:
        return start, (today + timedelta(days=30)).isoformat()

    if not start and not end:
        return today.isoformat(), (today + timedelta(days=7)).isoformat()

    return start, end


def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        return {
            "active_window": {},
            "tasks": [],
            "medication": [],
            "do_not_disturb": [],
            "exclusions": []
        }


# =========================
# NORMALIZATION
# =========================
def normalize_tasks(parsed):
    tasks = parsed.get("tasks") or []
    out = []

    for t in tasks:
        if not isinstance(t, dict):
            continue

        out.append({
            "type": t.get("type"),
            "interval_minutes": safe_int(t.get("interval_minutes")),
            "start_time": parse_time(t.get("start_time")),
            "end_time": parse_time(t.get("end_time"))
        })
    return out


def normalize_medication(parsed):
    meds = parsed.get("medication") or []
    out = []

    for m in meds:
        doses = []
        for t in (m.get("times") or []):
            pt = parse_time(t)
            if pt:
                doses.append({"h": pt[0], "m": pt[1]})

        if not doses:
            continue

        start, end = resolve_dates(m.get("start"), m.get("end"))

        out.append({
            "label": m.get("label", "Medication"),
            "start": start,
            "end": end,
            "days": normalize_days(m.get("days")),
            "doses": doses
        })

    return out


def build_plan(parsed):
    return {
        "tasks": normalize_tasks(parsed),
        "medication": normalize_medication(parsed),
        "dnd": parsed.get("do_not_disturb", []),
        "active_window": parsed.get("active_window", {})
    }


# =========================
# CONVERTER (FIXED)
# =========================
def convert_to_device_schema(plan):

    config = json.loads(json.dumps(BASE_CONFIG))
    config["_meta"]["ts_written"] = int(datetime.now(IST).timestamp())

    active = plan.get("active_window") or {}
    global_start = parse_time(active.get("start"))
    global_end = parse_time(active.get("end"))

    # ✅ TASKS RESTORED
    for t in plan.get("tasks", []):
        start = t["start_time"] or global_start
        end = t["end_time"] or global_end

        sh, sm = start if start else (0, 0)
        eh, em = end if end else (23, 59)

        if t["type"] == "hydration":
            config["hydration"].update({
                "enabled": True,
                "interval_ms": (t["interval_minutes"] or 30) * 60000,
                "start_hour": sh,
                "start_min": sm,
                "end_hour": eh,
                "end_min": em
            })

        elif t["type"] == "eye":
            config["eye"].update({
                "enabled": True,
                "interval_ms": (t["interval_minutes"] or 20) * 60000,
                "start_hour": sh,
                "start_min": sm,
                "end_hour": eh,
                "end_min": em
            })

        elif t["type"] == "stretch":
            config["stretch"].update({
                "enabled": True,
                "interval_ms": (t["interval_minutes"] or 60) * 60000,
                "phases": [{
                    "sh": sh,
                    "sm": sm,
                    "eh": eh,
                    "em": em
                }]
            })

        elif t["type"] == "walk":
            config["walk"].update({
                "enabled": True,
                "interval_min": t["interval_minutes"] or 120,
                "start_hour": sh,
                "start_min": sm,
                "end_hour": eh,
                "end_min": em
            })

    # ✅ MEDICATION
    if plan.get("medication"):
        config["medication_cfg"]["enabled"] = True
        config["medication"] = plan["medication"]

    # ✅ DND
    dnd = plan.get("dnd") or []
    if dnd:
        s = parse_time(dnd[0].get("start"))
        e = parse_time(dnd[0].get("end"))
        if s and e:
            config["dnd"].update({
                "enabled": True,
                "sh": s[0],
                "sm": s[1],
                "eh": e[0],
                "em": e[1]
            })

    return config


# =========================
# ROUTES
# =========================
@app.route("/parse", methods=["POST"])
def parse_schedule():
    logs = []

    try:
        data = request.get_json()
        logs.append("Received input")

        response = client.responses.create(
            model="gpt-5-nano",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": data["text"]}
            ]
        )

        raw = response.output_text
        logs.append(f"LLM output: {raw[:200]}")

        parsed = safe_json_parse(raw)
        plan = build_plan(parsed)
        config = convert_to_device_schema(plan)

        return jsonify({"data": config, "logs": logs})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/")
def home():
    return "Adaptive Scheduler Engine 🚀"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)