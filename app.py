import os
import json
import time
import re
import copy
from datetime import datetime, date, timedelta

import pytz
from dotenv import load_dotenv
from flask_cors import CORS
from flask import Flask, request, jsonify
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
app.json.sort_keys = False

frontend_origin = os.environ.get("FRONTEND_ORIGIN", "*")

CORS(
    app,
    resources={r"/*": {"origins": frontend_origin}},
    supports_credentials=True
)

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

IST = pytz.timezone("Asia/Kolkata")


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are a strict scheduling assistant.

Return ONLY valid JSON. No explanation text.


TYPO TOLERANCE:

- Users often misspell words on mobile keyboards or voice
  transcription (e.g. "remaind"/"remind me", "recouring"/
  "recurring", "wenesday"/"wednesday", "toursday"/"thursday",
  "everydey"/"every day", "hydartion"/"hydration"). Silently
  interpret the intended word from context and proceed normally.
  Never reject input, never ask about a typo, never reproduce
  the typo back to the user — just understand it. (Missing
  times are handled by the server, not by you.)

CASUAL CONVERSATION MODE:

- If the user's message is a greeting, small talk, a thank-you,
  a farewell, or a question about you ("what is frost", "who are
  you", "what can you do", "good morning") and it contains NO
  scheduling or reminder instructions, do NOT use the schedule
  format below.

- Instead return ONLY:

{
  "chat_reply": "short, warm, 1-3 sentence reply in plain English"
}

- When relevant, mention that you are Frost, a scheduling
  assistant that can set up hydration, eye-rest, stretch, walk,
  meditation, pomodoro, bottle-cleaning or medication reminders,
  and can also track the user's own custom habits (for example
  playing chess or reading a book) with start and end dates.

- Keep the tone friendly and casual, not robotic. Vary the
  wording naturally instead of repeating the same stock line.

- If the message mixes small talk WITH an actual scheduling
  request (e.g. "hey good morning, remind me to drink water
  every hour"), ignore this section entirely and use the normal
  schedule format below for the scheduling part.


SUPPORTED TASK TYPES:

hydration, eye, stretch, walk, meditation, pomodoro, bottle_clean

For anything that is NOT one of the above and is not a
medication (e.g. "remind me to call mom", "remind me to submit
the assignment"), use the separate CUSTOM HABIT / REMINDER RULES below
instead — do not force it into one of these task types.


ABSOLUTE MODE:

- If user gives one or more specific/exact clock times
  (e.g. "at 9am", "at 8am exactly", "at 9am and 1pm",
  "at 10am 11am 2pm 3pm 5am"):

    - set "times" to EVERY exact time the user listed, in the
      same order, for example:
      ["10:00","11:00","14:00","15:00","05:00"]

    - A single "at" can introduce a list of times. The following
      times may be separated by spaces, commas, semicolons, or "and".
    - Support at least 12 explicit absolute time values.
    - NEVER keep only the first time when multiple exact times
      were supplied.
    - DO NOT include interval_minutes
    - DO NOT include start_time / end_time for this task
    - This applies even if only ONE exact time is given.

Example:

User: "i want to drink water at 8am exactly"

Correct task:

{
  "type": "hydration",
  "times": ["08:00"]
}

WRONG (never do this):

{
  "type": "hydration",
  "start_time": "08:00",
  "end_time": "08:00"
}


INTERVAL MODE:

- If user says "every X minutes" or "every X hours":

    - set interval_minutes

    - start_time/end_time describe the WINDOW during which it
      repeats (they must NOT be equal to each other unless the
      user explicitly gave a window that small).
    - Overnight windows crossing midnight (e.g. "from 11pm to 3am")
      are fully valid.


ACTIVE WINDOW RULES:

- active_window is the user's overall active/working hours for
  the WHOLE message, not a single reminder's exact time.

- Never set active_window.start equal to active_window.end.

- If the user only gives one exact time for one reminder and
  does not mention working hours at all, leave active_window as
  the default full day ("00:00" to "23:59").


Never include both interval_minutes and times.


DAY RULES:

- Every task (hydration, eye, stretch, walk, meditation, pomodoro)
  and every medication entry supports an optional "days" list using
  these lowercase 3-letter codes only:

  ["mon","tue","wed","thu","fri","sat","sun"]

- If the user names specific day(s) for a task
  (e.g. "on sunday", "on mon and wed", "every tuesday"),
  set "days" to ONLY those days for that task.

- "weekdays" ->
  ["mon","tue","wed","thu","fri"]

- "weekends" ->
  ["sat","sun"]

- If the user does not mention any day at all, or says "daily" /
  "every day", OMIT the "days" field entirely (it will default
  to every day).


MEDICATION FORMAT & UPDATES:

{
  "label": "string",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "duration_days": int,
  "days": ["mon","tue","wed","thu","fri","sat","sun"],
  "times": ["HH:MM","HH:MM"],
  "action": "set" | "disable" | "remove"
}

- When modifying an existing medication dose (e.g. "actually make the morning dose 9am instead of 8am"),
  USE THE EXISTING MEDICATION LABEL. Do NOT create a new medication called "Morning dose".
- For global cancellation ("cancel all medications", "turn off all medication reminders"), set:
  { "label": "all medication", "action": "disable" }


CUSTOM HABIT / REMINDER RULES:

- Put each one in the top-level "custom" array (NOT in "tasks").
- "label" is a short activity name in plain words WITHOUT trailing category nouns
  (e.g. "Reading", "Morning Exercise", NOT "Reading habit").
- For global cancellation ("delete all habits", "turn off all custom habits"), set:
  { "label": "all habits", "action": "disable" }


ACTION RULES:

    "action": "set"      (create or update)
    "action": "disable"  (turn off / stop / pause / cancel)
    "action": "remove"   (delete completely)
"""

MAX_ABSOLUTE_TIMES = 12
DEFAULT_DURATION_DAYS = 7
DURATION_COUNTS_TODAY = False

TASK_TYPES = {
    "hydration",
    "eye",
    "stretch",
    "walk",
    "meditation",
    "pomodoro",
    "bottle_clean"
}

ABSOLUTE_TASK_TYPES = {
    "hydration",
    "eye",
    "stretch",
    "walk",
    "bottle_clean"
}

DISPLAY_NAMES = {
    "hydration": "Hydration reminder",
    "eye": "Eye break",
    "stretch": "Stretch break",
    "walk": "Walk",
    "meditation": "Meditation",
    "pomodoro": "Pomodoro",
    "bottle_clean": "Bottle cleaning"
}

ALL_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
WEEKDAYS = ["mon", "tue", "wed", "thu", "fri"]
WEEKEND = ["sat", "sun"]

DAY_MAP = {
    "monday": "mon", "mon": "mon",
    "tuesday": "tue", "tues": "tue", "tue": "tue",
    "wednesday": "wed", "wed": "wed",
    "thursday": "thu", "thurs": "thu", "thur": "thu", "thu": "thu",
    "friday": "fri", "fri": "fri",
    "saturday": "sat", "sat": "sat",
    "sunday": "sun", "sun": "sun"
}


# =========================================================
# HELPERS
# =========================================================

def safe_int(val):
    try:
        return int(val)
    except Exception:
        return None

def parse_time(t):
    try:
        h, m = t.split(":")
        return int(h), int(m)
    except Exception:
        return None

def window_duration_minutes(sh, sm, eh, em, allow_overnight=True):
    """
    Minutes between a start and end clock time.
    If allow_overnight is True and end < start, computes overnight span (+1440 min).
    """
    try:
        start_min = int(sh) * 60 + int(sm)
        end_min = int(eh) * 60 + int(em)
        if start_min == end_min:
            return 0
        if end_min < start_min:
            if allow_overnight:
                return (end_min + 1440) - start_min
            return None
        return end_min - start_min
    except (TypeError, ValueError):
        return None

def normalize_days(days):
    if not isinstance(days, list) or not days:
        return ALL_DAYS[:]
    valid = {d for d in days if d in ALL_DAYS}
    if not valid:
        return ALL_DAYS[:]
    return [d for d in ALL_DAYS if d in valid]

def today_ist():
    return datetime.now(IST).date()

def parse_iso_date(value):
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value.strip())
    except Exception:
        return None

def positive_int(value, maximum=3650):
    n = safe_int(value)
    if n is None or n <= 0:
        return None
    return min(n, maximum)

def nth_active_day(first, n, days):
    active = set(days or ALL_DAYS)
    d = first
    count = 0
    for _ in range(n * 7 + 14):
        if ALL_DAYS[d.weekday()] in active:
            count += 1
            if count == n:
                return d
        d += timedelta(days=1)
    return d

def span_end(start, n, days=None):
    first = start if DURATION_COUNTS_TODAY else start + timedelta(days=1)
    return nth_active_day(first, n, days)

def norm_days_opt(days):
    if not isinstance(days, list) or not days:
        return None
    return normalize_days(days)

def normalize_action(value):
    v = str(value or "set").strip().lower()
    if v in ("remove", "delete", "clear"):
        return "remove"
    if v in ("disable", "off", "stop", "pause", "cancel", "turn_off"):
        return "disable"
    return "set"

def clean_label(value, default=None):
    s = re.sub(r"\s+", " ", str(value or "")).strip()
    if not s:
        return default
    # Strip trailing category nouns (e.g., "Reading habit" -> "Reading")
    s_stripped = re.sub(r"\s+(?:habit|routine|reminder|task|medication|pill)s?$", "", s, flags=re.IGNORECASE).strip()
    if s_stripped:
        s = s_stripped
    return s[0].upper() + s[1:] if s else default

def _norm_label(s):
    s = re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower())
    s = re.sub(r"\s+", " ", s).strip()
    # Strip trailing category words for fuzzy matching
    s_clean = re.sub(r"\s+(?:habit|routine|reminder|task|medication|pill)s?$", "", s).strip()
    return s_clean or s

def find_by_label(items, label):
    n = _norm_label(label)
    if not n:
        return None
    for it in items:
        if _norm_label(it.get("label")) == n:
            return it
    for it in items:
        e = _norm_label(it.get("label"))
        if len(e) >= 3 and len(n) >= 3 and (n in e or e in n):
            return it
    return None

def next_item_id(items, prefix):
    highest = 0
    for it in items:
        m = re.search(r"(\d+)$", str(it.get("id", "")))
        if m:
            highest = max(highest, int(m.group(1)))
    return f"{prefix}_{highest + 1:03d}"

def parse_time_list(values):
    out = []
    seen = set()
    for ts in values or []:
        pt = parse_time(ts) if isinstance(ts, str) else None
        if not pt:
            continue
        if not (0 <= pt[0] <= 23 and 0 <= pt[1] <= 59):
            continue
        if pt in seen:
            continue
        seen.add(pt)
        out.append({"h": pt[0], "m": pt[1]})
    return out

def merge_time_lists(old, new):
    out = []
    seen = set()
    for t in list(old or []) + list(new or []):
        key = (t.get("h"), t.get("m"))
        if key in seen:
            continue
        seen.add(key)
        out.append({"h": t["h"], "m": t["m"]})
    out.sort(key=lambda t: (t["h"], t["m"]))
    return out

def fmt_time12(h, m):
    suffix = "AM" if h < 12 else "PM"
    return f"{(h % 12) or 12}:{m:02d} {suffix}"

def fmt_times(times):
    return ", ".join(fmt_time12(t["h"], t["m"]) for t in times)

def fmt_date(value):
    d = parse_iso_date(value)
    return d.strftime("%d %b %Y") if d else str(value)

def fmt_days(days):
    days = normalize_days(days)
    if days == ALL_DAYS:
        return "every day"
    if days == WEEKDAYS:
        return "weekdays"
    if days == WEEKEND:
        return "weekends"
    return ", ".join(d.capitalize() for d in days)

def extract_volume_phrase(text):
    if not isinstance(text, str):
        return None
    m = re.search(r"\b(\d+(?:\.\d+)?\s*(?:ml|milliliters?|l|liters?|litres?|glasses?|bottles?))\b", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------
# Duration / date-range parsing
# ---------------------------------------------------------

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20, "thirty": 30
}

_DURATION_RE = re.compile(
    r"(?:\b(?:for|of|over|within|next)\s+(?:the\s+)?(?:next\s+)?"
    r"(?P<n1>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve|fifteen|twenty|thirty)"
    r"|(?<!every )\b(?P<n2>\d+))"
    r"\s*-?\s*(?P<unit>days?|weeks?|months?)\b"
    r"(?!\s+(?:a|per|each|every)\s+(?:week|month|day)\b)",
    re.IGNORECASE
)

_MONTHS = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)

_WEEKDAY_NAMES = (
    r"(?:monday|mon|tuesday|tues|tue|wednesday|wed|thursday|thurs|"
    r"thur|thu|friday|fri|saturday|sat|sunday|sun)"
)

_DATE_PATTERNS = [
    _DURATION_RE,
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"),
    re.compile(
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+" + _MONTHS
        + r"\b(?:\s*,?\s*\d{4})?",
        re.IGNORECASE
    ),
    re.compile(
        r"\b" + _MONTHS + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?\b"
        r"(?:\s*,?\s*\d{4})?",
        re.IGNORECASE
    ),
    re.compile(
        r"\b(?:until|till|upto|up\s+to|through|thru)\s+"
        r"(?:next\s+|this\s+)?" + _WEEKDAY_NAMES + r"\b",
        re.IGNORECASE
    )
]

def find_duration_phrases(text):
    out = []
    if not isinstance(text, str):
        return out
    for m in _DURATION_RE.finditer(text):
        raw = (m.group("n1") or m.group("n2") or "").lower()
        n = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw)
        if not n:
            continue
        unit = m.group("unit").lower()
        mult = (
            30 if unit.startswith("month")
            else 7 if unit.startswith("week")
            else 1
        )
        out.append(n * mult)
    return out

def strip_date_phrases(text):
    if not isinstance(text, str):
        return text
    for pat in _DATE_PATTERNS:
        text = pat.sub(" ", text)
    return text

def apply_duration_fallback(parsed, user_text):
    phrases = find_duration_phrases(user_text)
    if len(phrases) != 1:
        return parsed
    days = phrases[0]
    for key in ("tasks", "medication", "custom"):
        for item in parsed.get(key) or []:
            if not isinstance(item, dict):
                continue
            if normalize_action(item.get("action")) != "set":
                continue
            if item.get("date"):
                continue
            if any(
                item.get(k)
                for k in (
                    "start_date", "end_date", "start", "end",
                    "duration_days", "extend_days"
                )
            ):
                continue
            item["duration_days"] = days
    return parsed

def date_fields(item):
    return {
        "start_date": parse_iso_date(
            item.get("start_date") or item.get("start")
        ),
        "end_date": parse_iso_date(
            item.get("end_date") or item.get("end")
        ),
        "duration_days": positive_int(item.get("duration_days")),
        "extend_days": positive_int(item.get("extend_days"))
    }

def resolve_item_dates(item, existing=None):
    today = today_ist()
    days = item.get("days")
    if days:
        days = normalize_days(days)
    elif existing and existing.get("enabled") and existing.get("days"):
        days = normalize_days(existing.get("days"))
    else:
        days = ALL_DAYS[:]

    ex_start = ex_end = None
    if existing and existing.get("enabled"):
        ex_start = parse_iso_date(
            existing.get("start_date") or existing.get("start")
        )
        ex_end = parse_iso_date(
            existing.get("end_date") or existing.get("end")
        )
        if ex_end and ex_end < today:
            ex_start = ex_end = None

    start = item.get("start_date")
    end = item.get("end_date")
    n = item.get("duration_days")
    ext = item.get("extend_days")

    used_default = False

    if ext:
        base_end = ex_end if ex_end else today
        start = start or ex_start or today
        end = nth_active_day(base_end + timedelta(days=1), ext, days)
    elif n:
        start = start or today
        end = end or span_end(start, n, days)
    elif start and not end:
        end = (
            ex_end
            if ex_end and ex_end >= start
            else span_end(start, DEFAULT_DURATION_DAYS)
        )
    elif end and not start:
        start = ex_start if ex_start and ex_start <= end else today
    elif not start and not end:
        if ex_start and ex_end:
            start, end = ex_start, ex_end
        else:
            start = today
            end = span_end(today, DEFAULT_DURATION_DAYS)
            used_default = True

    if end < start:
        return None, None, False, (
            f"end date {fmt_date(end)} is before the start date {fmt_date(start)}"
        )
    if end < today:
        return None, None, False, (
            f"end date {fmt_date(end)} is already in the past"
        )
    return start, end, used_default, None


# =========================================================
# SAFE JSON PARSE
# =========================================================

def safe_json_parse(text):
    try:
        data = json.loads(text)
        if "active_window" not in data or not data["active_window"]:
            data["active_window"] = {"start": "00:00", "end": "23:59"}
        if not data["active_window"].get("start"):
            data["active_window"]["start"] = "00:00"
        if not data["active_window"].get("end"):
            data["active_window"]["end"] = "23:59"
        if data["active_window"].get("start") == data["active_window"].get("end"):
            data["active_window"]["start"] = "00:00"
            data["active_window"]["end"] = "23:59"
        return data
    except Exception:
        return {
            "active_window": {"start": "00:00", "end": "23:59"},
            "tasks": [],
            "medication": [],
            "custom": [],
            "do_not_disturb": [],
            "exclusions": []
        }


# =========================================================
# TEXT EXTRACTION
# =========================================================

def _parse_time_token(hour_str, minute_str, ampm):
    if len(hour_str) > 2:
        return None, f"'{hour_str}' is not a valid hour"
    hour = int(hour_str)
    minute = 0
    if minute_str is not None:
        if len(minute_str) > 2:
            return None, f"'{minute_str}' is not a valid minute"
        minute = int(minute_str)

    if minute > 59:
        return None, f"minute {minute:02d} is out of range (00-59)"

    if ampm:
        if not 1 <= hour <= 12:
            return None, f"hour {hour} is invalid with am/pm (must be 1-12)"
        if ampm.lower() == "am":
            hour24 = 0 if hour == 12 else hour
        else:
            hour24 = 12 if hour == 12 else hour + 12
    else:
        if hour > 23:
            return None, f"hour {hour} is out of range (00-23)"
        hour24 = hour

    return (hour24, minute), None

def extract_explicit_times_from_text(text):
    if not isinstance(text, str):
        return [], []

    text_lower = text.lower()
    time_token = re.compile(
        r"(\d{1,4})(?:\s*[:.]{1,2}\s*(\d{1,4}))?\s*(am|pm)?"
        r"(?:\s+exactly)?",
        re.IGNORECASE
    )
    separator = re.compile(r"\s*(?:(?:,|;|and)\s*)?", re.IGNORECASE)

    at_matches = list(re.finditer(r"\bat\b", text_lower))
    if not at_matches:
        return [], []

    found = []
    invalid = []

    for at_match in at_matches:
        pos = at_match.end()
        while pos < len(text_lower):
            sep_match = separator.match(text_lower, pos)
            if sep_match:
                pos = sep_match.end()

            match = time_token.match(text_lower, pos)
            if not match or not match.group(1):
                break

            token_text = match.group(0).strip()
            pos = match.end()

            value, reason = _parse_time_token(
                match.group(1),
                match.group(2),
                match.group(3)
            )

            if value is None:
                invalid.append({"token": token_text, "reason": reason})
                continue

            if value in found:
                continue

            if len(found) >= MAX_ABSOLUTE_TIMES:
                invalid.append({
                    "token": token_text,
                    "reason": f"only the first {MAX_ABSOLUTE_TIMES} absolute times are kept"
                })
                continue

            found.append(value)

    return found, invalid

_DAY_TOKEN = (
    r"(?:monday|mon|tuesday|tues|tue|wednesday|wed|thursday|thurs|"
    r"thur|thu|friday|fri|saturday|sat|sunday|sun|weekdays?|weekends?)s?"
)

_EXCLUDE_RE = re.compile(
    r"\b(?:do\s+not|don't|dont|not|except|excluding|exclude|skip|"
    r"without|no)\s+"
    r"(?:(?:to|rema?ind\w*|disturb\w*|notify|alert|me|on|for|the|any|"
    r"of|every|days?)\s+)*"
    r"(" + _DAY_TOKEN + r"(?:(?:\s*[,&]\s*|\s+(?:and|or)\s+|\s+)"
    + _DAY_TOKEN + r")*)\b",
    re.IGNORECASE
)

def _expand_day_token(tok):
    tok = tok.lower()
    if tok.startswith("weekday"):
        return WEEKDAYS[:]
    if tok.startswith("weekend"):
        return WEEKEND[:]
    if tok not in DAY_MAP and tok.endswith("s"):
        tok = tok[:-1]
    code = DAY_MAP.get(tok)
    return [code] if code else []

def extract_days_from_text(text):
    if not isinstance(text, str):
        return None
    text_lower = text.lower()
    excluded = set()

    def _grab(m):
        for tok in re.findall(_DAY_TOKEN, m.group(1)):
            excluded.update(_expand_day_token(tok))
        return " "

    text_lower = _EXCLUDE_RE.sub(_grab, text_lower)
    positive = set()

    if re.search(r"\bweekdays?\b", text_lower):
        positive.update(WEEKDAYS)
    elif re.search(r"\bweekends?\b", text_lower):
        positive.update(WEEKEND)
    else:
        pattern = re.compile(
            r"\b(monday|mon|tuesday|tues|tue|wednesday|wed|"
            r"thursday|thurs|thur|thu|friday|fri|saturday|sat|"
            r"sunday|sun)\b",
            re.IGNORECASE
        )
        for match in pattern.finditer(text_lower):
            code = DAY_MAP.get(match.group(1).lower())
            if code:
                positive.add(code)

    if positive:
        result = [d for d in ALL_DAYS if d in positive and d not in excluded]
    elif excluded:
        result = [d for d in ALL_DAYS if d not in excluded]
    else:
        return None

    return result or None

def force_absolute_times_from_user_text(parsed, user_text):
    explicit_times, invalid_tokens = extract_explicit_times_from_text(user_text)

    if invalid_tokens:
        parsed["_invalid_times"] = [
            {"task": "absolute_time", "reason": f"\"{item['token']}\" — {item['reason']}"}
            for item in invalid_tokens
        ]

    if not explicit_times:
        return parsed

    tasks = parsed.get("tasks") or []
    for task in tasks:
        if task.get("type") not in ABSOLUTE_TASK_TYPES:
            continue
        if normalize_action(task.get("action")) != "set":
            continue
        task["times"] = [f"{h:02d}:{m:02d}" for h, m in explicit_times]
        task.pop("interval_minutes", None)
        task.pop("start_time", None)
        task.pop("end_time", None)

    if not parsed.get("active_window"):
        parsed["active_window"] = {"start": "00:00", "end": "23:59"}

    return parsed

def apply_medication_extras(plan, user_text):
    stated_snooze = extract_stated_snooze_minutes(user_text)
    if stated_snooze is None:
        return plan
    for med in plan.get("medication", []) or []:
        med["snooze_min"] = stated_snooze
    return plan

def force_window_from_text(parsed, user_text):
    if not isinstance(user_text, str):
        return parsed

    pattern = re.compile(
        r"from\s+(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?\s+to\s+"
        r"(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?",
        re.IGNORECASE
    )
    match = pattern.search(user_text)
    if not match:
        return parsed

    def to_24h(hour, minute, ampm):
        hour = int(hour)
        minute = int(minute or 0)
        if ampm:
            ampm = ampm.lower()
            if ampm == "am":
                hour = 0 if hour == 12 else hour
            else:
                hour = 12 if hour == 12 else hour + 12
        return hour, minute

    ap1 = match.group(3)
    ap2 = match.group(6)
    sh, sm = to_24h(match.group(1), match.group(2), ap1 or ap2)
    eh, em = to_24h(match.group(4), match.group(5), ap2 or ap1)

    window_types = {"hydration", "eye", "stretch", "walk"}
    for task in parsed.get("tasks", []) or []:
        if task.get("type") in window_types:
            task["start_time"] = f"{sh:02d}:{sm:02d}"
            task["end_time"] = f"{eh:02d}:{em:02d}"
            task.pop("times", None)

    return parsed

def extract_stated_duration_minutes(text):
    if not isinstance(text, str):
        return None
    match = re.search(r"\b(\d+)\s*(?:minutes?|mins?)\b", text.lower())
    if not match:
        return None
    prefix = text.lower()[:match.start()].rstrip()
    if prefix.endswith("every"):
        return None
    return int(match.group(1))

def extract_stated_snooze_minutes(text):
    if not isinstance(text, str):
        return None
    t = text.lower()
    patterns = [
        r"(\d+)\s*(?:minutes?|mins?)\s*snooze",
        r"snooze\s*(?:period\s*)?(?:of\s*)?(\d+)\s*(?:minutes?|mins?)",
        r"(\d+)\s*(?:minutes?|mins?)\s*re(?:mainder|minder)\s*period",
        r"re(?:mainder|minder)\s*period\s*(?:of\s*)?(\d+)\s*(?:minutes?|mins?)"
    ]
    for pattern in patterns:
        match = re.search(pattern, t)
        if match:
            return int(match.group(1))
    return None

def extract_medication_window(text):
    if not isinstance(text, str):
        return None
    pattern = re.compile(
        r"from\s+(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?\s*to\s*"
        r"(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?",
        re.IGNORECASE
    )
    match = pattern.search(text)
    if not match:
        return None

    def to_24h(hour, minute, ampm):
        hour = int(hour)
        minute = int(minute or 0)
        if ampm:
            ampm = ampm.lower()
            if ampm == "am":
                hour = 0 if hour == 12 else hour
            else:
                hour = 12 if hour == 12 else hour + 12
        return hour, minute

    ap1 = match.group(3)
    ap2 = match.group(6)
    sh, sm = to_24h(match.group(1), match.group(2), ap1 or ap2)
    eh, em = to_24h(match.group(4), match.group(5), ap2 or ap1)
    return (sh, sm, eh, em)

def extract_stated_gap_minutes(text):
    if not isinstance(text, str):
        return None
    match = re.search(r"(\d+)\s*(?:minutes?|mins?)\s*gap", text.lower())
    if match:
        return int(match.group(1))
    match = re.search(r"gap\s*(?:of\s*)?(\d+)\s*(?:minutes?|mins?)", text.lower())
    if match:
        return int(match.group(1))
    return None

def check_medication_mismatch(plan, user_text):
    window = extract_medication_window(user_text)
    gap = extract_stated_gap_minutes(user_text)
    if not window or gap is None:
        return []

    sh, sm, eh, em = window
    span = window_duration_minutes(sh, sm, eh, em)
    if not span:
        return []

    mismatches = []
    for med in plan.get("medication", []) or []:
        doses = med.get("doses", [])
        count = len(doses)
        if count < 2:
            continue
        needed = gap * (count - 1)
        if needed > span:
            mismatches.append({
                "task": "medication",
                "reason": (
                    f"medication '{med.get('label', 'Medication')}': "
                    f"{count} doses with a {gap} min gap need {needed} min, "
                    f"but the stated window {sh:02d}:{sm:02d}→{eh:02d}:{em:02d} is only {span} min"
                )
            })
            med["_invalid"] = True
    return mismatches

def check_duration_mismatch(plan, user_text):
    stated = extract_stated_duration_minutes(user_text)
    if stated is None:
        return []

    mismatches = []
    for t in plan.get("tasks", []):
        # Disambiguate per-session duration from overall window span
        if t.get("interval_minutes") is not None:
            continue
        if re.search(r"\bevery\b", user_text, re.IGNORECASE):
            continue

        start = t.get("start_time")
        end = t.get("end_time")
        if not start or not end:
            continue

        span = window_duration_minutes(start[0], start[1], end[0], end[1], allow_overnight=False)
        if span and span > 0 and span != stated and span < 240:
            mismatches.append({
                "task": t.get("type"),
                "reason": (
                    f"{t.get('type')}: stated duration {stated} min "
                    f"does not match window {start[0]:02d}:{start[1]:02d}→{end[0]:02d}:{end[1]:02d} ({span} min)"
                )
            })

    return mismatches

def apply_days_from_user_text(parsed, user_text):
    extracted = extract_days_from_text(user_text)
    if not extracted:
        return parsed
    for task in parsed.get("tasks", []) or []:
        if not task.get("days"):
            task["days"] = extracted
    for med in parsed.get("medication", []) or []:
        if not med.get("days"):
            med["days"] = extracted
    return parsed


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_tasks(parsed):
    tasks = parsed.get("tasks") or []
    out = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        task_type = t.get("type")
        if task_type not in TASK_TYPES:
            continue

        parsed_times = parse_time_list(t.get("times"))
        start_time = parse_time(t.get("start_time"))
        end_time = parse_time(t.get("end_time"))
        interval_minutes = safe_int(t.get("interval_minutes"))

        if (
            task_type in ABSOLUTE_TASK_TYPES
            and not parsed_times
            and start_time
            and not end_time
            and interval_minutes is None
        ):
            parsed_times = [{"h": start_time[0], "m": start_time[1]}]
            start_time = None

        if (
            task_type in ABSOLUTE_TASK_TYPES
            and not parsed_times
            and start_time
            and end_time
            and start_time == end_time
            and interval_minutes is None
        ):
            parsed_times = [{"h": start_time[0], "m": start_time[1]}]
            start_time = None
            end_time = None

        mode = "absolute" if parsed_times else "interval"

        entry = {
            "type": task_type,
            "action": normalize_action(t.get("action")),
            "mode": mode,
            "interval_minutes": interval_minutes,
            "interval_days": positive_int(t.get("interval_days"), maximum=365),
            "times": parsed_times,
            "merge_times": bool(t.get("merge_times")),
            "start_time": start_time,
            "end_time": end_time,
            "focus_min": safe_int(t.get("focus_min")),
            "break_min": safe_int(t.get("break_min")),
            "cycles": safe_int(t.get("cycles")),
            "days": norm_days_opt(t.get("days"))
        }
        entry.update(date_fields(t))
        out.append(entry)
    return out

def normalize_medication(parsed):
    meds = parsed.get("medication") or []
    out = []
    for m in meds:
        if not isinstance(m, dict):
            continue
        entry = {
            "label": clean_label(m.get("label"), "Medication"),
            "new_label": clean_label(m.get("new_label")),
            "action": normalize_action(m.get("action")),
            "doses": parse_time_list(m.get("times")),
            "merge_times": bool(m.get("merge_times")),
            "days": norm_days_opt(m.get("days"))
        }
        entry.update(date_fields(m))
        out.append(entry)
    return out

def normalize_custom(parsed):
    items = parsed.get("custom") or []
    out = []
    for c in items:
        if not isinstance(c, dict):
            continue
        times = parse_time_list(c.get("times"))
        entry = {
            "label": clean_label(c.get("label"), "Reminder"),
            "new_label": clean_label(c.get("new_label")),
            "action": normalize_action(c.get("action")),
            "times": times,
            "merge_times": bool(c.get("merge_times")),
            "days": norm_days_opt(c.get("days")),
            "one_time_date": parse_iso_date(c.get("date"))
        }
        entry.update(date_fields(c))
        out.append(entry)
    return out

def build_plan(parsed):
    active = parsed.get("active_window") or {}
    return {
        "tasks": normalize_tasks(parsed),
        "medication": normalize_medication(parsed),
        "custom": normalize_custom(parsed),
        "dnd": parsed.get("do_not_disturb", []),
        "active_window": {
            "start": active.get("start", "00:00"),
            "end": active.get("end", "23:59")
        }
    }


# =========================================================
# V6 CONFIG - BASE TEMPLATE
# =========================================================

CUSTOM_TEXT_DEFAULTS = {
    "text_x": 120,
    "text_y": 100,
    "text_size": 1,
    "text_color": 65535,
    "text_align": 1,
    "text_width": 180
}

MEDICATION_TEXT_DEFAULTS = {
    "text_x": 120,
    "text_y": 135,
    "text_size": 1,
    "text_color": 65535,
    "text_align": 1,
    "text_width": 180
}

COUNTER_DEFAULTS = {
    "x": 118,
    "y": 105,
    "text_size": 1,
    "text_color": 65535,
    "text_align": 1
}

def make_base_config():
    today = today_ist()
    s = today.isoformat()
    e = span_end(today, DEFAULT_DURATION_DAYS).isoformat()

    def basic():
        return {
            "enabled": False,
            "start_date": s,
            "end_date": e,
            "days": ALL_DAYS[:],
            "abs": {"times": []},
            "display_ms": 60000,
            "require_ack": True
        }

    return {
        "_meta": {
            "schema_ver": 6,
            "device": "FROST"
        },
        "reminders": {
            "hydration": basic(),
            "stretch": basic(),
            "eye": basic(),
            "walk": basic(),
            "bottle_clean": {
                "enabled": False,
                "start_date": s,
                "end_date": e,
                "interval_days": 1,
                "time": {"h": 18, "m": 0},
                "display_ms": 60000,
                "require_ack": True
            },
            "meditation": {
                "enabled": False,
                "start_date": s,
                "end_date": e,
                "days": ALL_DAYS[:],
                "times": [],
                "display_ms": 600000,
                "require_ack": True
            },
            "medication": {
                "enabled": False,
                "require_ack": True,
                "snooze_min": 15,
                "display_ms": 60000,
                "medicines": []
            },
            "custom": {
                "enabled": False,
                "require_ack": True,
                "display_ms": 60000,
                "events": []
            }
        },
        "audio": {
            "volume": 15,
            "pomodoro": {
                "enabled": True,
                "tracks": [45]
            },
            "meditation": {
                "enabled": True,
                "tracks": [45]
            },
            "healing": {
                "enabled": True,
                "require_dock": True,
                "tracks": [45]
            },
            "healing_schedules": [
                {
                    "enabled": True,
                    "start_time": "11:00",
                    "end_time": "12:00",
                    "days": ALL_DAYS[:6]
                },
                {
                    "enabled": True,
                    "start_time": "16:00",
                    "end_time": "17:00",
                    "days": ALL_DAYS[:6]
                }
            ]
        },
        "pomodoro": {
            "enabled": False,
            "start_date": s,
            "end_date": e,
            "days": ALL_DAYS[:],
            "focus_min": 25,
            "break_min": 5,
            "laps": [],
            "focus_counter": dict(COUNTER_DEFAULTS),
            "break_counter": dict(COUNTER_DEFAULTS)
        }
    }

def load_working_config(current):
    base = make_base_config()
    if not (
        isinstance(current, dict)
        and isinstance(current.get("reminders"), dict)
    ):
        return base

    cfg = copy.deepcopy(current)
    legacy = cfg.pop("bottle_clean", None)
    if isinstance(legacy, dict) and "bottle_clean" not in cfg["reminders"]:
        bottle = copy.deepcopy(base["reminders"]["bottle_clean"])
        bottle["enabled"] = bool(legacy.get("enabled"))
        bottle["interval_days"] = legacy.get("interval_days", 1)
        bottle["time"] = {
            "h": legacy.get("hour", 18),
            "m": legacy.get("minute", 0)
        }
        cfg["reminders"]["bottle_clean"] = bottle

    for k, v in base.items():
        if k not in cfg:
            cfg[k] = copy.deepcopy(v)

    for k, v in base["reminders"].items():
        if k not in cfg["reminders"]:
            cfg["reminders"][k] = copy.deepcopy(v)

    cfg["_meta"] = base["_meta"]
    cfg["reminders"]["medication"].setdefault("medicines", [])
    cfg["reminders"]["custom"].setdefault("events", [])
    return cfg


# =========================================================
# V6 CONVERTER
# =========================================================

class Ctx:
    def __init__(self, config, global_start, global_end, user_text=""):
        self.config = config
        self.reminders = config["reminders"]
        self.gs = global_start
        self.ge = global_end
        self.user_text = user_text
        self.invalid = []
        self.missing = []
        self.summary = []
        self.notes = []

    def need_time(self, task, question):
        self.missing.append({
            "task": task,
            "question": question
        })

    def default_note(self, name, end_date):
        self.notes.append(
            f"No duration was given for {name}, so I set it for "
            f"{DEFAULT_DURATION_DAYS} days (until "
            f"{fmt_date(end_date)}). Tell me a number of days or an "
            f"end date if you want something different."
        )

def has_schedule(cur):
    return bool(cur.get("enabled") and (cur.get("abs") or {}).get("times"))

def expand_interval_times(sh, sm, eh, em, interval_min):
    out = []
    start_min = sh * 60 + sm
    end_min = eh * 60 + em
    if end_min < start_min:
        end_min += 1440

    cur = start_min
    while cur <= end_min:
        norm_m = cur % 1440
        out.append({"h": norm_m // 60, "m": norm_m % 60})
        cur += interval_min
    return out

def _ask_basic(ctx, key):
    name = DISPLAY_NAMES[key].lower()
    ctx.need_time(
        key,
        f"What time should I set for your {name}? Give exact times "
        f"(e.g. \"at 10:15am and 2pm\") or an interval "
        f"(e.g. \"every 30 minutes from 9am to 5pm\")."
    )

def _apply_simple_task(ctx, t, key, default_interval_min):
    cur = ctx.reminders[key]
    name = DISPLAY_NAMES[key]

    if t["action"] in ("disable", "remove"):
        cur["enabled"] = False
        ctx.summary.append(f"{name}: turned off")
        return

    start_d, end_d, used_default, err = resolve_item_dates(t, cur)
    if err:
        ctx.invalid.append({"task": key, "reason": f"{name}: {err}"})
        return

    days = t.get("days") or (
        normalize_days(cur.get("days"))
        if cur.get("enabled")
        else ALL_DAYS[:]
    )

    sh, sm = t.get("start_time") or ctx.gs or (0, 0)
    eh, em = t.get("end_time") or ctx.ge or (23, 59)

    wants_interval = (
        t.get("interval_minutes") is not None
        or t.get("start_time") is not None
        or t.get("end_time") is not None
    )

    if t.get("times"):
        times = t["times"]
        if t.get("merge_times"):
            times = merge_time_lists((cur.get("abs") or {}).get("times"), times)

        # Enforce 12-reminder limit: ask the user without automatically changing schedule
        if len(times) > MAX_ABSOLUTE_TIMES:
            ctx.need_time(
                key,
                f"Setting {name.lower()} with {len(times)} times exceeds the limit of "
                f"{MAX_ABSOLUTE_TIMES} reminders. Please reduce the number of times."
            )
            return

        what = fmt_times(times)

    elif wants_interval:
        span = window_duration_minutes(sh, sm, eh, em, allow_overnight=True)
        interval_min = t.get("interval_minutes") or default_interval_min

        if span is None or span <= 0:
            ctx.invalid.append({
                "task": key,
                "reason": f"{key} window {sh:02d}:{sm:02d}→{eh:02d}:{em:02d} is zero-length"
            })
            return

        if interval_min > span:
            ctx.invalid.append({
                "task": key,
                "reason": f"{key}: every {interval_min} min doesn't fit in the {span}-min window"
            })
            return

        times = expand_interval_times(sh, sm, eh, em, interval_min)

        # 12-reminder limit: ask user to increase interval or shorten window
        if len(times) > MAX_ABSOLUTE_TIMES:
            ctx.need_time(
                key,
                f"Setting {name.lower()} every {interval_min} minutes between "
                f"{fmt_time12(sh, sm)} and {fmt_time12(eh, em)} would create {len(times)} reminders, "
                f"but at most {MAX_ABSOLUTE_TIMES} exact times are allowed. "
                f"Please increase the interval (e.g. every 60 minutes) or shorten the time window."
            )
            return

        what = f"{fmt_times(times)} (every {interval_min} min)"

    elif has_schedule(cur):
        times = (cur.get("abs") or {}).get("times")
        what = fmt_times(times)

    else:
        _ask_basic(ctx, key)
        return

    new = {
        "enabled": True,
        "start_date": start_d.isoformat(),
        "end_date": end_d.isoformat(),
        "days": days,
        "abs": {"times": times},
        "display_ms": cur.get("display_ms", 60000),
        "require_ack": cur.get("require_ack", True)
    }

    ctx.reminders[key] = new

    # Acknowledge target volume if present
    volume = extract_volume_phrase(ctx.user_text)
    if key == "hydration" and volume:
        ctx.summary.append(
            f"{name} ({volume}): {what} · {fmt_days(days)} · "
            f"{fmt_date(start_d)} → {fmt_date(end_d)}"
        )
        ctx.notes.append(f"Target volume: {volume}.")
    else:
        ctx.summary.append(
            f"{name}: {what} · {fmt_days(days)} · "
            f"{fmt_date(start_d)} → {fmt_date(end_d)}"
        )

    if used_default:
        ctx.default_note(name, end_d)

def _apply_meditation(ctx, med_tasks):
    cur = ctx.reminders["meditation"]
    name = DISPLAY_NAMES["meditation"]
    active = [t for t in med_tasks if t["action"] == "set"]

    if not active:
        cur["enabled"] = False
        ctx.summary.append(f"{name}: turned off")
        return

    first = active[0]
    start_d, end_d, used_default, err = resolve_item_dates(first, cur)
    if err:
        ctx.invalid.append({"task": "meditation", "reason": f"{name}: {err}"})
        return

    windows = []
    spans = []
    bad_window = False

    for t in active:
        if not (t.get("start_time") and t.get("end_time")):
            continue
        sh, sm = t["start_time"]
        eh, em = t["end_time"]

        # For meditation, end < start without explicit overnight is reversed/invalid
        span = window_duration_minutes(sh, sm, eh, em, allow_overnight=False)
        if span is None or span <= 0 or span > 720:
            bad_window = True
            ctx.invalid.append({
                "task": "meditation",
                "reason": f"meditation window {sh:02d}:{sm:02d}→{eh:02d}:{em:02d} is zero-length or reversed"
            })
            continue

        windows.append({
            "start": {"h": sh, "m": sm},
            "end": {"h": eh, "m": em}
        })
        spans.append(span)

    existing_windows = cur.get("times") if cur.get("enabled") else []

    if windows:
        if first.get("merge_times"):
            windows = (existing_windows or []) + windows
        if len(windows) > MAX_ABSOLUTE_TIMES:
            ctx.need_time("meditation", f"Setting meditation with {len(windows)} sessions exceeds the limit of {MAX_ABSOLUTE_TIMES} sessions. Please reduce the number of sessions.")
            return
        display_ms = max(spans) * 60000
    elif bad_window:
        return
    elif existing_windows:
        windows = existing_windows
        display_ms = cur.get("display_ms", 600000)
    else:
        ctx.need_time(
            "meditation",
            "What start and end time should I set for meditation? (e.g. \"from 7am to 7:30am\")"
        )
        return

    days = first.get("days") or (
        normalize_days(cur.get("days")) if cur.get("enabled") else ALL_DAYS[:]
    )

    ctx.reminders["meditation"] = {
        "enabled": True,
        "start_date": start_d.isoformat(),
        "end_date": end_d.isoformat(),
        "days": days,
        "times": windows,
        "display_ms": display_ms,
        "require_ack": cur.get("require_ack", True)
    }

    what = ", ".join(
        f"{fmt_time12(w['start']['h'], w['start']['m'])}–{fmt_time12(w['end']['h'], w['end']['m'])}"
        for w in windows
    )
    ctx.summary.append(
        f"{name}: {what} · {fmt_days(days)} · {fmt_date(start_d)} → {fmt_date(end_d)}"
    )

    if used_default:
        ctx.default_note(name, end_d)

def _apply_pomodoro(ctx, t):
    pomo = ctx.config["pomodoro"]
    name = DISPLAY_NAMES["pomodoro"]

    if t["action"] in ("disable", "remove"):
        pomo["enabled"] = False
        ctx.summary.append(f"{name}: turned off")
        return

    start_d, end_d, used_default, err = resolve_item_dates(t, pomo)
    if err:
        ctx.invalid.append({"task": "pomodoro", "reason": f"{name}: {err}"})
        return

    was_on = bool(pomo.get("enabled"))
    focus = t.get("focus_min") or (pomo.get("focus_min") if was_on else None) or 25
    brk = t.get("break_min") or (pomo.get("break_min") if was_on else None) or 5

    # Negative or zero durations validation
    if focus <= 0 or brk <= 0:
        ctx.invalid.append({
            "task": "pomodoro",
            "reason": "pomodoro focus and break durations must be positive numbers"
        })
        return

    starts = []
    if t.get("start_time"):
        starts.append(t["start_time"])
    for tm in t.get("times") or []:
        pair = (tm["h"], tm["m"])
        if pair not in starts:
            starts.append(pair)

    existing_laps = pomo.get("laps") if was_on else []
    cycles = t.get("cycles")

    if starts:
        if t.get("start_time") and t.get("end_time"):
            sh, sm = t["start_time"]
            eh, em = t["end_time"]
            span = window_duration_minutes(sh, sm, eh, em, allow_overnight=False)
            if span is None or span <= 0:
                ctx.invalid.append({
                    "task": "pomodoro",
                    "reason": f"pomodoro window {sh:02d}:{sm:02d}→{eh:02d}:{em:02d} is zero-length or reversed"
                })
                return

            fit = (span + brk) // (focus + brk)
            if cycles is None:
                if fit < 1:
                    ctx.invalid.append({
                        "task": "pomodoro",
                        "reason": f"pomodoro: a {focus}-min focus session doesn't fit in window"
                    })
                    return
                cycles = fit
            else:
                needed = focus * cycles + brk * max(cycles - 1, 0)
                if needed > span:
                    ctx.invalid.append({
                        "task": "pomodoro",
                        "reason": f"pomodoro: {cycles} cycles need {needed} min, but window is {span} min"
                    })
                    return

        if cycles is None:
            cycles = (existing_laps[0].get("cycles") if existing_laps else 4) or 4

        laps = [{"start": {"h": h, "m": m}, "cycles": cycles} for h, m in starts]
        if len(laps) > MAX_ABSOLUTE_TIMES:
            ctx.need_time("pomodoro", f"Setting pomodoro with {len(laps)} start times exceeds the limit of {MAX_ABSOLUTE_TIMES}. Please reduce the number of start times.")
            return
    elif existing_laps:
        laps = existing_laps
    else:
        ctx.need_time("pomodoro", "What time should the pomodoro start? (e.g. \"at 9am\")")
        return

    days = t.get("days") or (
        normalize_days(pomo.get("days")) if was_on else ALL_DAYS[:]
    )

    new = {
        "enabled": True,
        "start_date": start_d.isoformat(),
        "end_date": end_d.isoformat(),
        "days": days,
        "focus_min": focus,
        "break_min": brk,
        "laps": laps,
        "focus_counter": pomo.get("focus_counter", dict(COUNTER_DEFAULTS)),
        "break_counter": pomo.get("break_counter", dict(COUNTER_DEFAULTS))
    }

    for k, v in pomo.items():
        if k not in new and k not in ("cycles", "lap_mode_enabled", "auto_start_break", "auto_start_focus"):
            new[k] = v

    ctx.config["pomodoro"] = new
    what = ", ".join(f"{fmt_time12(l['start']['h'], l['start']['m'])} ({l['cycles']} cycles)" for l in laps)
    ctx.summary.append(f"{name} {focus}/{brk}: {what} · {fmt_days(days)} · {fmt_date(start_d)} → {fmt_date(end_d)}")

    if used_default:
        ctx.default_note(name, end_d)

def _apply_bottle_clean(ctx, t):
    cur = ctx.reminders["bottle_clean"]
    name = DISPLAY_NAMES["bottle_clean"]

    if t["action"] in ("disable", "remove"):
        cur["enabled"] = False
        ctx.summary.append(f"{name}: turned off")
        return

    start_d, end_d, used_default, err = resolve_item_dates(t, cur)
    if err:
        ctx.invalid.append({"task": "bottle_clean", "reason": f"{name}: {err}"})
        return

    was_on = bool(cur.get("enabled"))
    if t.get("times"):
        time_obj = dict(t["times"][0])
    elif was_on and cur.get("time"):
        time_obj = cur["time"]
    else:
        ctx.need_time("bottle_clean", "What time of day should I remind you to clean your bottle? (e.g. \"at 6pm\")")
        return

    interval_days = t.get("interval_days") or (cur.get("interval_days") if was_on else None) or 1
    new = {
        "enabled": True,
        "start_date": start_d.isoformat(),
        "end_date": end_d.isoformat(),
        "interval_days": interval_days,
        "time": time_obj,
        "display_ms": cur.get("display_ms", 60000),
        "require_ack": cur.get("require_ack", True)
    }
    for k, v in cur.items():
        if k not in new:
            new[k] = v

    ctx.reminders["bottle_clean"] = new
    how_often = "every day" if interval_days == 1 else f"every {interval_days} days"
    ctx.summary.append(f"{name}: {fmt_time12(time_obj['h'], time_obj['m'])} · {how_often} · {fmt_date(start_d)} → {fmt_date(end_d)}")

    if used_default:
        ctx.default_note(name, end_d)

def _apply_medication(ctx, plan_meds):
    med_cfg = ctx.reminders["medication"]
    medicines = med_cfg.setdefault("medicines", [])

    for m in plan_meds:
        if m.get("_invalid"):
            continue

        label = m["label"]
        label_norm = _norm_label(label)

        # Global cancellation check (e.g. "cancel all medication reminders")
        raw_label = str(label or "").lower().strip()
        is_all_cancel = (
            label_norm in (
                "all", "all medication", "all medications", "medication", "medications",
                "all medicine", "all medicines", "medicine", "medicines", "pills", "all pills"
            )
            or raw_label in (
                "all", "all medication", "all medications", "medication", "medications",
                "all medicine", "all medicines", "medicine", "medicines", "pills", "all pills",
                "all reminders", "all meds", "all pills"
            )
            or "all med" in raw_label or "all pill" in raw_label
        )
        if m["action"] in ("disable", "remove") and is_all_cancel:
            if m["action"] == "remove":
                medicines.clear()
                med_cfg["enabled"] = False
                ctx.summary.append("All medication reminders removed")
            else:
                for med_item in medicines:
                    med_item["enabled"] = False
                med_cfg["enabled"] = False
                ctx.summary.append("All medication reminders turned off")
            continue

        match = find_by_label(medicines, label)

        # Multi-turn relative update safety net: if only 1 active medicine exists,
        # relative label like "Morning dose" maps to that medicine
        active_meds = [x for x in medicines if x.get("enabled") is not False]
        if not match and len(active_meds) == 1:
            if re.search(r"\b(dose|morning|evening|night|afternoon)\b", label_norm) or len(plan_meds) == 1:
                match = active_meds[0]

        if m["action"] in ("disable", "remove"):
            if not match:
                ctx.invalid.append({"task": "medication", "reason": f"no medication named '{label}' found"})
                continue
            if m["action"] == "remove":
                medicines.remove(match)
                ctx.summary.append(f"Medication '{match['label']}': removed")
            else:
                match["enabled"] = False
                ctx.summary.append(f"Medication '{match['label']}': turned off")
            continue

        start_d, end_d, used_default, err = resolve_item_dates(m, match)
        if err:
            ctx.invalid.append({"task": "medication", "reason": f"medication '{label}': {err}"})
            continue

        doses = m["doses"]
        if match and match.get("doses"):
            if m.get("merge_times"):
                doses = merge_time_lists(match.get("doses"), doses)
            elif doses and len(doses) < len(match.get("doses", [])):
                # Updating one dose (e.g. morning dose 8am -> 9am)
                old_doses = list(match.get("doses", []))
                new_d = doses[0]
                # Replace the closest/morning dose
                if len(old_doses) == 2:
                    if new_d["h"] < 14: # Morning dose update
                        old_doses[0] = new_d
                    else:
                        old_doses[1] = new_d
                    doses = old_doses

        if not doses:
            if match and match.get("doses"):
                doses = match["doses"]
            else:
                ctx.need_time("medication", f"What time(s) should I remind you to take {label}? (e.g. \"at 8am and 8pm\")")
                continue

        if len(doses) > MAX_ABSOLUTE_TIMES:
            ctx.need_time("medication", f"Setting medication '{label}' with {len(doses)} doses exceeds the limit of {MAX_ABSOLUTE_TIMES} reminders. Please reduce the number of doses.")
            continue

        days = m.get("days") or (normalize_days(match.get("days")) if match else ALL_DAYS[:])

        if match:
            # Keep original label if new label was a generic dose descriptor
            new_lbl = m.get("new_label")
            if not new_lbl and not re.search(r"\b(dose|morning|evening)\b", label_norm):
                new_lbl = m["label"]
            match.update({
                "label": new_lbl or match.get("label"),
                "enabled": True,
                "start": start_d.isoformat(),
                "end": end_d.isoformat(),
                "days": days,
                "doses": doses
            })
            shown = match["label"]
        else:
            medicines.append({
                "id": next_item_id(medicines, "med"),
                "label": label,
                "enabled": True,
                "start": start_d.isoformat(),
                "end": end_d.isoformat(),
                "days": days,
                **MEDICATION_TEXT_DEFAULTS,
                "doses": doses
            })
            shown = label

        ctx.summary.append(
            f"Medication '{shown}': {fmt_times(doses)} · {fmt_days(days)} · {fmt_date(start_d)} → {fmt_date(end_d)}"
        )
        if used_default:
            ctx.default_note(f"medication '{shown}'", end_d)
        if m.get("snooze_min") is not None:
            med_cfg["snooze_min"] = m["snooze_min"]

    med_cfg["enabled"] = any(x.get("enabled") for x in medicines)

def _apply_custom(ctx, plan_custom):
    cust = ctx.reminders["custom"]
    events = cust.setdefault("events", [])

    for c in plan_custom:
        label = c["label"]
        label_norm = _norm_label(label)

        # Global cancellation check (e.g. "cancel all habits")
        raw_label = str(label or "").lower().strip()
        is_all_cancel = (
            label_norm in ("all", "all habits", "habits", "habit", "all custom habits", "all reminders", "reminders", "custom")
            or raw_label in ("all", "all habits", "habits", "habit", "all custom habits", "all reminders", "reminders", "custom", "cancel all", "all routines")
            or "all habit" in raw_label or "all reminder" in raw_label
        )
        if c["action"] in ("disable", "remove") and is_all_cancel:
            if c["action"] == "remove":
                events.clear()
                cust["enabled"] = False
                ctx.summary.append("All habits removed")
            else:
                for ev in events:
                    ev["enabled"] = False
                cust["enabled"] = False
                ctx.summary.append("All habits turned off")
            continue

        match = find_by_label(events, label)

        if c["action"] in ("disable", "remove"):
            if not match:
                ctx.invalid.append({"task": "custom", "reason": f"no habit named '{label}' found"})
                continue
            if c["action"] == "remove":
                events.remove(match)
                ctx.summary.append(f"Habit '{match['label']}': removed")
            else:
                match["enabled"] = False
                ctx.summary.append(f"Habit '{match['label']}': turned off")
            continue

        times = c["times"]
        if times and match and c.get("merge_times"):
            times = merge_time_lists(match.get("times"), times)

        if not times:
            if match and match.get("times"):
                times = match["times"]
            else:
                ctx.need_time("custom", f"What time should I remind you for \"{label}\"? (e.g. \"at 6pm\")")
                continue

        if len(times) > MAX_ABSOLUTE_TIMES:
            ctx.need_time("custom", f"Setting habit '{label}' with {len(times)} times exceeds the limit of {MAX_ABSOLUTE_TIMES} reminders. Please reduce the number of times.")
            continue

        used_default = False
        if c.get("one_time_date"):
            one = c["one_time_date"]
            if one < today_ist():
                ctx.invalid.append({"task": "custom", "reason": f"habit '{label}': {fmt_date(one)} is already in the past"})
                continue
            start_d = end_d = one
            days = ALL_DAYS[:]
        else:
            start_d, end_d, used_default, err = resolve_item_dates(c, match)
            if err:
                ctx.invalid.append({"task": "custom", "reason": f"habit '{label}': {err}"})
                continue
            days = c.get("days") or (normalize_days(match.get("days")) if match else ALL_DAYS[:])

        if match:
            match.update({
                "label": c.get("new_label") or match.get("label"),
                "enabled": True,
                "start_date": start_d.isoformat(),
                "end_date": end_d.isoformat(),
                "days": days,
                "times": times
            })
            shown = match["label"]
        else:
            events.append({
                "id": next_item_id(events, "custom"),
                "label": label,
                "enabled": True,
                "start_date": start_d.isoformat(),
                "end_date": end_d.isoformat(),
                "display_ms": cust.get("display_ms", 60000),
                "days": days,
                "times": times,
                **CUSTOM_TEXT_DEFAULTS
            })
            shown = label

        when = f"{fmt_date(start_d)} (one time)" if start_d == end_d else f"{fmt_date(start_d)} → {fmt_date(end_d)}"
        ctx.summary.append(f"Habit '{shown}': {fmt_times(times)} · {fmt_days(days)} · {when}")
        if used_default:
            ctx.default_note(f"'{shown}'", end_d)

    cust["enabled"] = any(e.get("enabled") for e in events)

def convert_to_new_schema(plan, current_config=None, user_text=""):
    config = load_working_config(current_config)
    active = plan.get("active_window") or {"start": "00:00", "end": "23:59"}
    ctx = Ctx(
        config,
        parse_time(active.get("start") or "00:00"),
        parse_time(active.get("end") or "23:59"),
        user_text=user_text
    )

    meditation_tasks = []
    for t in plan.get("tasks", []):
        tt = t["type"]
        if tt == "hydration":
            _apply_simple_task(ctx, t, "hydration", 30)
        elif tt == "eye":
            _apply_simple_task(ctx, t, "eye", 20)
        elif tt == "stretch":
            _apply_simple_task(ctx, t, "stretch", 60)
        elif tt == "walk":
            _apply_simple_task(ctx, t, "walk", 120)
        elif tt == "meditation":
            meditation_tasks.append(t)
        elif tt == "pomodoro":
            _apply_pomodoro(ctx, t)
        elif tt == "bottle_clean":
            _apply_bottle_clean(ctx, t)

    if meditation_tasks:
        _apply_meditation(ctx, meditation_tasks)

    _apply_medication(ctx, plan.get("medication", []))
    _apply_custom(ctx, plan.get("custom", []))

    return (
        config,
        ctx.invalid,
        ctx.missing,
        ctx.summary,
        ctx.notes
    )


# =========================================================
# FOLLOW-UP QUESTIONS & CANCEL
# =========================================================

CANCEL_RE = re.compile(
    r"^\s*(?:cancel|never\s*mind|nevermind|forget\s+it|"
    r"skip(?:\s+it)?|no\s+thanks?)\s*[.!]*\s*$",
    re.IGNORECASE
)

def merge_pending(text, pending):
    text = (text or "").strip()
    if not (isinstance(pending, dict) and pending.get("text")):
        return text, False
    if CANCEL_RE.match(text):
        return text, True

    answer = text
    if re.match(r"^\d", answer) and not re.search(r"\bat\b", answer, re.IGNORECASE):
        answer = "at " + answer

    combined = f"{str(pending['text']).strip()}. {answer}"
    return combined[:2000], False

def build_question_reply(missing):
    if len(missing) == 1:
        return missing[0]["question"]
    lines = [f"{i}. {m['question']}" for i, m in enumerate(missing, 1)]
    return "I need a little more information before I can save this:\n" + "\n".join(lines)


# =========================================================
# ROUTE
# =========================================================

@app.route("/parse", methods=["POST"])
def parse_schedule():
    logs = []
    start_time = time.perf_counter()

    try:
        data = request.get_json() or {}
        logs.append("Step 1: Input received")

        user_text, cancelled = merge_pending(
            data.get("text", ""),
            data.get("pending")
        )

        if cancelled:
            logs.append("Step 1b: Pending request cancelled by user")
            return jsonify({
                "reply": "No problem - I've dropped that request.",
                "logs": logs
            })

        if data.get("pending"):
            logs.append("Step 1b: Follow-up answer merged with earlier request")

        today_ist_dt = datetime.now(IST)
        dated_system_prompt = (
            SYSTEM_PROMPT
            + "\n\nTODAY'S DATE: "
            + today_ist_dt.strftime("%Y-%m-%d (%A)")
        )

        # Context enrichment: inform LLM of existing active medicines and habits
        active_context = []
        if data.get("current_config"):
            cur_rem = data["current_config"].get("reminders", {})
            cur_meds = [m for m in cur_rem.get("medication", {}).get("medicines", []) if m.get("enabled") is not False]
            if cur_meds:
                active_context.append("Active Medications: " + ", ".join(f"'{m.get('label')}'" for m in cur_meds))
            cur_habits = [h for h in cur_rem.get("custom", {}).get("events", []) if h.get("enabled") is not False]
            if cur_habits:
                active_context.append("Active Custom Habits: " + ", ".join(f"'{h.get('label')}'" for h in cur_habits))

        if active_context:
            dated_system_prompt += "\n\n" + "\n".join(active_context)

        raw = None
        if client:
            try:
                response = client.responses.create(
                    model="gpt-5-nano",
                    input=[
                        {"role": "system", "content": dated_system_prompt},
                        {"role": "user", "content": user_text}
                    ]
                )
                raw = response.output_text
            except Exception as llm_err:
                logs.append(f"LLM API notice: {llm_err}")

        # If LLM unavailable or offline fallback
        if not raw:
            raw = "{}"

        logs.append("Step 2: LLM response received")
        logs.append(f"Preview: {raw[:120]}...")

        parsed = safe_json_parse(raw)

        if isinstance(parsed, dict) and "chat_reply" in parsed:
            logs.append("Step 2b: Casual conversation detected, skipping schedule parsing")
            elapsed = (time.perf_counter() - start_time) * 1000
            logs.append(f"⏱ Total time: {round(elapsed, 2)} ms")
            return jsonify({
                "reply": parsed["chat_reply"],
                "logs": logs
            })

        time_text = strip_date_phrases(user_text)

        parsed = force_absolute_times_from_user_text(parsed, time_text)
        time_parse_invalid = parsed.pop("_invalid_times", [])

        parsed = force_window_from_text(parsed, time_text)
        parsed = apply_days_from_user_text(parsed, time_text)
        parsed = apply_duration_fallback(parsed, user_text)

        logs.append("Step 3: JSON parsed, exact-time, day and duration rules applied")

        plan = build_plan(parsed)
        plan = apply_medication_extras(plan, time_text)

        if plan.get("active_window", {}).get("start") == plan.get("active_window", {}).get("end"):
            plan["active_window"] = {"start": "00:00", "end": "23:59"}

        logs.append(
            f"Step 4: Tasks={len(plan['tasks'])}, Meds={len(plan['medication'])}, Habits={len(plan['custom'])}"
        )

        duration_mismatches = check_duration_mismatch(plan, time_text)
        medication_mismatches = check_medication_mismatch(plan, time_text)

        config, invalid, missing, summary, notes = convert_to_new_schema(
            plan,
            data.get("current_config"),
            user_text=user_text
        )

        # 12-reminder limit or missing parameters -> ask user without changing schedule
        if missing:
            for item in missing:
                logs.append(f"❓ Need more info ({item['task']}): {item['question']}")

            elapsed = (time.perf_counter() - start_time) * 1000
            logs.append(f"⏱ Total time: {round(elapsed, 2)} ms")
            return jsonify({
                "reply": build_question_reply(missing),
                "needs_input": True,
                "missing": missing,
                "pending": {"text": user_text},
                "logs": logs
            })

        invalid = time_parse_invalid + duration_mismatches + medication_mismatches + invalid

        for item in duration_mismatches:
            task_cfg = config.get("reminders", {}).get(item["task"])
            if task_cfg is not None:
                task_cfg["enabled"] = False

        logs.append("Step 5: Device config generated (v6 schema)")

        if invalid:
            for item in invalid:
                logs.append(f"⚠ Rejected (invalid input): {item['reason']}")

        reminders_cfg = config.get("reminders", {})
        enabled = [
            k for k in ["hydration", "eye", "stretch", "walk", "bottle_clean", "meditation", "custom"]
            if reminders_cfg.get(k, {}).get("enabled")
        ]
        if config.get("pomodoro", {}).get("enabled"):
            enabled.append("pomodoro")

        logs.append("Enabled: " + (", ".join(enabled) if enabled else "none"))

        medicines = reminders_cfg.get("medication", {}).get("medicines", [])
        if medicines:
            logs.append(f"Medication count: {len(medicines)}")

        custom_events = reminders_cfg.get("custom", {}).get("events", [])
        if custom_events:
            logs.append(f"Custom habit count: {len(custom_events)}")

        elapsed = (time.perf_counter() - start_time) * 1000
        logs.append(f"⏱ Total time: {round(elapsed, 2)} ms")

        return jsonify({
            "data": config,
            "active_window": plan.get("active_window"),
            "invalid": invalid,
            "summary": summary,
            "notes": notes,
            "logs": logs
        })

    except Exception as e:
        elapsed = (time.perf_counter() - start_time) * 1000
        logs.append(f"ERROR: {str(e)}")
        logs.append(f"⏱ Total time: {round(elapsed, 2)} ms")
        return jsonify({"error": str(e), "logs": logs}), 400


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return "Adaptive Scheduler Engine 🚀"


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
