import os
import json
import time
import re
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

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Add it to your .env "
        "file (OPENAI_API_KEY=sk-...) in the project root."
    )

client = OpenAI(api_key=api_key)

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
  Never reject input, never ask for clarification, never
  reproduce the typo back to the user — just understand it.

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
  meditation, pomodoro, or medication reminders.

- Keep the tone friendly and casual, not robotic. Vary the
  wording naturally instead of repeating the same stock line.

- If the message mixes small talk WITH an actual scheduling
  request (e.g. "hey good morning, remind me to drink water
  every hour"), ignore this section entirely and use the normal
  schedule format below for the scheduling part.


SUPPORTED TASK TYPES:

hydration, eye, stretch, walk, meditation, pomodoro

For anything that is NOT one of the above and is not a
medication (e.g. "remind me to call mom", "remind me to submit
the assignment"), use the separate CUSTOM REMINDER RULES below
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

- If user says "every X minutes":

    - set interval_minutes

    - start_time/end_time describe the WINDOW during which it
      repeats (they must NOT be equal to each other unless the
      user explicitly gave a window that small).


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

- If a day phrase appears once in the message but applies to the
  whole schedule (e.g. "meditate from 1pm to 1:15pm on sunday"),
  attach "days" to the specific task(s) it modifies, not to
  unrelated tasks that clearly have their own separate timing.


Example:

User: "meditate from 1pm to 1:15pm on sunday"

Correct task:

{
  "type": "meditation",
  "start_time": "13:00",
  "end_time": "13:15",
  "days": ["sun"]
}


Example:

User: "drink water every 30 minutes on weekdays"

Correct task:

{
  "type": "hydration",
  "interval_minutes": 30,
  "days": ["mon","tue","wed","thu","fri"]
}


MEDITATION RULES:

- Meditation is a time block (start → end)

- Example:
  "meditate from 10am to 10:15am"

- Return:

{
  "type": "meditation",
  "start_time": "HH:MM",
  "end_time": "HH:MM",
  "days": ["mon","tue","wed","thu","fri","sat","sun"]
}

- Only include "days" if the user specified day(s).


POMODORO RULES:

- Example:
  "pomodoro 25/5 for 4 cycles from 9am to 12pm"

- Extract:

{
  "type": "pomodoro",
  "focus_min": int,
  "break_min": int,
  "cycles": int,
  "start_time": "HH:MM",
  "end_time": "HH:MM"
}

- If cycles missing -> default 4


IMPORTANT:

- If NO explicit pomodoro time window is provided:

    - use lap_mode_enabled = true

    - use active_window as lap window

- If explicit pomodoro timing is provided:

    - use lap_mode_enabled = false

    - do NOT create laps


MEDICATION FORMAT:

{
  "label": "string",
  "start": "YYYY-MM-DD",
  "end": "YYYY-MM-DD",
  "days": ["mon","tue","wed","thu","fri","sat","sun"],
  "times": ["HH:MM","HH:MM"]
}

- "days" follows the same DAY RULES as tasks above:
  only the days the user actually named, or every day by default.


CUSTOM REMINDER RULES:

- Use this for any reminder that is NOT hydration, eye, stretch,
  walk, meditation, pomodoro, or medication — e.g. "remind me to
  call mom", "remind me to submit the assignment", "remind me to
  check the oven".

- Put each one in the top-level "custom" array (NOT in "tasks").

- ONE-TIME reminder — a specific calendar date, "today",
  "tomorrow", a single named upcoming weekday meant as one
  occurrence, or the word "once"/"one time":

{
  "label": "string describing what to remind about",
  "date": "YYYY-MM-DD",
  "times": ["HH:MM"]
}

  - Resolve relative dates ("tomorrow", "next friday") to a real
    YYYY-MM-DD date using TODAY'S DATE given below.
  - Do NOT include "days" for a one-time reminder.

- RECURRING reminder — a repeating day-of-week pattern such as
  "every monday", "on mondays and wednesdays", "daily",
  "every day", "weekdays", "weekends", or "remind me daily":

{
  "label": "string describing what to remind about",
  "days": ["mon","tue","wed","thu","fri","sat","sun"],
  "times": ["HH:MM"]
}

  - Follow the same DAY RULES as tasks above. If the user says
    "every day"/"daily" with no specific days, OMIT "days"
    entirely (defaults to every day).
  - Do NOT include "date" for a recurring reminder.

- A single "at" can introduce a list of times, same as ABSOLUTE
  MODE above (e.g. "remind me to take a break at 11am and 3pm").

Example:

User: "remind me on 10am every monday to submit assignment"

{
  "custom": [
    {
      "label": "submit assignment",
      "days": ["mon"],
      "times": ["10:00"]
    }
  ]
}

Example:

User: "remind me to call mom tomorrow at 6pm"

{
  "custom": [
    {
      "label": "call mom",
      "date": "<tomorrow's real date>",
      "times": ["18:00"]
    }
  ]
}


FINAL FORMAT:

{
  "active_window": {"start": "HH:MM", "end": "HH:MM"},
  "tasks": [],
  "medication": [],
  "custom": [],
  "do_not_disturb": [],
  "exclusions": []
}
"""


# =========================================================
# BASE CONFIG
# =========================================================

BASE_CONFIG = {
    "_meta": {
        "schema_ver": None,
        "device": "FROST",
        "ts_written": 0
    },

    "tone_mode": "professional",

    "ui": {
        "action_log": {
            "enabled": True,
            "show_ms": 3000
        }
    },

    "dfplayer": {
        "volume": 24,
        "boot_volume": 15,
        "night_volume": 8,
        "night_start_hour": 22,
        "night_end_hour": 7,
        "night_mode_enabled": False
    },

    "audio": {
        "pomo_focus_music_enabled": True,
        "pomo_focus_music_track": 101,
        "pomo_focus_music_loop": True,
        "meditation_music_enabled": True,
        "meditation_music_track": 31
    },

    "hydration": {
        "enabled": False,
        "mode": "interval",
        "interval_ms": 7200000,
        "prompt_duration_ms": 60000,
        "prompt_gap_ms": 600000,
        "require_ack": True,
        "goal_ml": 2000,
        "start_hour": 7,
        "start_min": 0,
        "end_hour": 22,
        "end_min": 0,
        "days": [],
        "abs": {
            "enabled": False,
            "times": []
        }
    },

    "eye": {
        "enabled": False,
        "mode": "interval",
        "interval_ms": 1800000,
        "require_ack": True,
        "start_hour": 8,
        "start_min": 0,
        "end_hour": 20,
        "end_min": 0,
        "days": [],
        "abs": {
            "enabled": False,
            "times": []
        }
    },

    "stretch": {
        "enabled": False,
        "mode": "interval",
        "interval_ms": 3600000,
        "duration_ms": 60000,
        "require_ack": True,
        "days": [],
        "phases": [],
        "abs": {
            "enabled": False,
            "times": []
        }
    },

    "walk": {
        "enabled": False,
        "mode": "interval",
        "interval_min": 120,
        "display_sec": 90,
        "require_ack": True,
        "start_hour": 8,
        "start_min": 0,
        "end_hour": 20,
        "end_min": 0,
        "days": [],
        "abs": {
            "enabled": False,
            "times": []
        }
    },

    "meditation": {
        "enabled": False,
        "sh": 0,
        "sm": 0,
        "eh": 0,
        "em": 0,
        "display_sec": 600,
        "days": [
            "mon",
            "tue",
            "wed",
            "thu",
            "fri",
            "sat",
            "sun"
        ]
    },

    "pomo": {
        "enabled": False,
        "focus_min": 25,
        "break_min": 5,
        "cycles": 4,
        "lap_mode_enabled": True,
        "laps": []
    },

    "dnd": {
        "enabled": False,
        "sh": 0,
        "sm": 0,
        "eh": 0,
        "em": 0,
        "allow_med": True,
        "allow_hydration": False,
        "allow_stretch": False,
        "allow_eye": False,
        "allow_cleaning": False,
        "allow_walk": False,
        "allow_meditation": False,
        "allow_healing": False,
        "allow_custom": False,
        "allow_pomodoro": False
    },

    "medication_cfg": {
        "enabled": False,
        "require_ack": True,
        "allow_device_ack": True,
        "snooze_min": 15,
        "default_window_min": 120,
        "show_ms": 60000
    },

    "medication": [],

    "custom": {
        "enabled": False,
        "require_ack": True,
        "snooze_min": 5,
        "events": []
    },

    "ack_config": {
        "force_mode": False
    },

    "custom_texts": {
        "hydration": "Time to drink water!",
        "stretch": "Time to stretch!",
        "eye": "Time for eye break!",
        "walk": "Time for a short walk!",
        "medication": "Medication reminder",
        "meditation": "Meditation time",
        "pomodoro_focus": "Focus time started",
        "pomodoro_break": "Break time started"
    },

    "images": {},

    "priority": []
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


def window_duration_minutes(sh, sm, eh, em):
    """
    Minutes between a start and end clock time on the same day.
    Zero or negative means the window is empty/reversed.
    """
    try:
        return (
            int(eh) * 60
            + int(em)
            - int(sh) * 60
            - int(sm)
        )
    except (TypeError, ValueError):
        return None


ALL_DAYS = [
    "mon",
    "tue",
    "wed",
    "thu",
    "fri",
    "sat",
    "sun"
]

WEEKDAYS = [
    "mon",
    "tue",
    "wed",
    "thu",
    "fri"
]

WEEKEND = [
    "sat",
    "sun"
]

DAY_MAP = {
    "monday": "mon",
    "mon": "mon",

    "tuesday": "tue",
    "tues": "tue",
    "tue": "tue",

    "wednesday": "wed",
    "wed": "wed",

    "thursday": "thu",
    "thurs": "thu",
    "thur": "thu",
    "thu": "thu",

    "friday": "fri",
    "fri": "fri",

    "saturday": "sat",
    "sat": "sat",

    "sunday": "sun",
    "sun": "sun"
}


def normalize_days(days):
    """
    Validate and canonically order a days list.
    Empty/missing/invalid means every day.
    """

    if not isinstance(days, list) or not days:
        return ALL_DAYS[:]

    valid = {
        d for d in days
        if d in ALL_DAYS
    }

    if not valid:
        return ALL_DAYS[:]

    return [
        d for d in ALL_DAYS
        if d in valid
    ]


def resolve_dates(start, end):
    today = date.today()

    if start and end:
        return start, end

    if not start and not end:
        return (
            today.isoformat(),
            (today + timedelta(days=7)).isoformat()
        )

    return start, end


# =========================================================
# SAFE JSON PARSE
# =========================================================

def safe_json_parse(text):
    try:
        data = json.loads(text)

        if "active_window" not in data or not data["active_window"]:
            data["active_window"] = {
                "start": "00:00",
                "end": "23:59"
            }

        if not data["active_window"].get("start"):
            data["active_window"]["start"] = "00:00"

        if not data["active_window"].get("end"):
            data["active_window"]["end"] = "23:59"

        if (
            data["active_window"].get("start")
            == data["active_window"].get("end")
        ):
            data["active_window"]["start"] = "00:00"
            data["active_window"]["end"] = "23:59"

        return data

    except Exception:
        return {
            "active_window": {
                "start": "00:00",
                "end": "23:59"
            },
            "tasks": [],
            "medication": [],
            "custom": [],
            "do_not_disturb": [],
            "exclusions": []
        }


# =========================================================
# TEXT EXTRACTION
# =========================================================

MAX_ABSOLUTE_TIMES = 12


def _parse_time_token(hour_str, minute_str, ampm):
    """
    Validate a single (hour, minute, am/pm) token and convert it to
    24h (hour, minute). Returns (value, reason) where value is None
    and reason is a human-readable rejection message when the token
    is not a real clock time.

    - Malformed digit runs (e.g. "111am", "222pm" parsed as a
      3-digit hour) are rejected outright.
    - Minutes/hours out of range (e.g. "11:99", "22pm") are
      rejected.
    - "." or ".." are accepted as a colon substitute so typos like
      "1.22" or "3..30" are read as "1:22" / "3:30" automatically.
    """

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
    """
    Extract exact clock times from an explicit "at" time list.

    Supports:
      "at 10am"
      "at 10am and 11am"
      "at 10am, 11am, 2pm, 3pm, 5am"
      "at 10am 11am 2pm 3pm 5am"
      "at 1.22"   -> 01:22 ("." accepted in place of ":")
      "at 3..30"  -> 03:30 (typo'd double-dot still accepted)

    The first "at" starts the list. Subsequent clock times in the
    same list do not need another "at". Up to MAX_ABSOLUTE_TIMES
    (12) valid times are kept, in the order given.

    A malformed or out-of-range token (e.g. "111am", "222pm",
    "11:99") does NOT abort the rest of the list -- it is skipped
    and reported, and parsing continues with whatever comes next.

    Returns a tuple: (valid_times, invalid_entries)
      - valid_times: list of (hour24, minute) tuples, deduped, in
        the order first seen, capped at MAX_ABSOLUTE_TIMES.
      - invalid_entries: list of {"token": str, "reason": str}
        for every token that looked like a time but wasn't valid.
    """

    if not isinstance(text, str):
        return [], []

    text_lower = text.lower()

    # A clock-time token. Minutes may be separated by ":" or by one
    # or two dots (covers "1.22" and the typo "3..30"). AM/PM is
    # optional so lists such as "at 10am 11am 2pm" are supported.
    time_token = re.compile(
        r"(\d{1,4})(?:\s*[:.]{1,2}\s*(\d{1,4}))?\s*(am|pm)?"
        r"(?:\s+exactly)?",
        re.IGNORECASE
    )

    separator = re.compile(
        r"\s*(?:(?:,|;|and)\s*)?",
        re.IGNORECASE
    )

    # "at" must explicitly introduce the absolute-time list.
    at_matches = list(re.finditer(r"\bat\b", text_lower))
    if not at_matches:
        return [], []

    found = []
    invalid = []

    # Parse each "at ..." clause independently. This also supports
    # messages containing more than one reminder clause.
    for at_match in at_matches:
        pos = at_match.end()

        while pos < len(text_lower):
            # Allow separators commonly used in a time list.
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
                invalid.append({
                    "token": token_text,
                    "reason": reason
                })
                # Keep scanning -- one bad token shouldn't drop the
                # rest of the list.
                continue

            if value in found:
                continue

            if len(found) >= MAX_ABSOLUTE_TIMES:
                invalid.append({
                    "token": token_text,
                    "reason": (
                        f"only the first {MAX_ABSOLUTE_TIMES} "
                        f"absolute times are kept"
                    )
                })
                continue

            found.append(value)

    return found, invalid


def extract_days_from_text(text):
    """
    Extract explicit day-of-week selections.
    """

    if not isinstance(text, str):
        return None

    text_lower = text.lower()

    if re.search(r"\bweekdays?\b", text_lower):
        return WEEKDAYS[:]

    if re.search(r"\bweekends?\b", text_lower):
        return WEEKEND[:]

    pattern = re.compile(
        r"\b(monday|mon|tuesday|tues|tue|wednesday|wed|"
        r"thursday|thurs|thur|thu|friday|fri|saturday|sat|"
        r"sunday|sun)\b",
        re.IGNORECASE
    )

    found = set()

    for match in pattern.finditer(text_lower):
        code = DAY_MAP.get(match.group(1).lower())

        if code:
            found.add(code)

    if not found:
        return None

    return [
        d for d in ALL_DAYS
        if d in found
    ]


def extract_water_goal_ml(text):
    """
    Detect a daily hydration goal.
    """

    if not isinstance(text, str):
        return None

    text_lower = text.lower()

    patterns = [
        r"\b(?:water\s+)?goal(?:\s+is|\s*:)?\s*"
        r"(\d+(?:\.\d+)?)\s*(ml|l|liters?|litres?)\b",

        r"\b(\d+(?:\.\d+)?)\s*(ml|l|liters?|litres?)\s+"
        r"(?:of\s+)?water\s+(?:today\s+)?(?:goal|target)\b",

        r"\b(?:today|daily)\s+(?:water\s+)?"
        r"(?:goal|target)\s*(?:is|of|:)?\s*"
        r"(\d+(?:\.\d+)?)\s*(ml|l|liters?|litres?)\b",

        r"\b(?:drink|have)\s+"
        r"(\d+(?:\.\d+)?)\s*(ml|l|liters?|litres?)\s+"
        r"(?:of\s+)?water\s+today\b",

        r"\b(?:drink|have)\s+"
        r"(\d+(?:\.\d+)?)\s*(ml|l|liters?|litres?)\s+"
        r"(?:of\s+)?water\b"
    ]

    for pattern in patterns:

        m = re.search(pattern, text_lower)

        if not m:
            continue

        value = float(m.group(1))
        unit = m.group(2)

        if unit == "ml":
            goal_ml = round(value)
        else:
            goal_ml = round(value * 1000)

        if 100 <= goal_ml <= 20000:
            return goal_ml

    return None


def apply_water_goal(parsed, user_text):

    goal_ml = extract_water_goal_ml(user_text)

    if goal_ml is not None:
        parsed["_hydration_goal_ml"] = goal_ml

    return parsed


def force_absolute_times_from_user_text(parsed, user_text):
    """
    Force supported reminder tasks into absolute mode when the
    user's original message explicitly contains 'at <time>'.

    Malformed/out-of-range tokens (e.g. "111am", "22pm", "11:99")
    are dropped individually and stashed on parsed["_invalid_times"]
    so the /parse route can surface them -- they no longer cause
    the rest of a valid time list to be discarded.
    """

    explicit_times, invalid_tokens = extract_explicit_times_from_text(
        user_text
    )

    if invalid_tokens:
        parsed["_invalid_times"] = [
            {
                "task": "absolute_time",
                "reason": f"\"{item['token']}\" — {item['reason']}"
            }
            for item in invalid_tokens
        ]

    if not explicit_times:
        return parsed

    tasks = parsed.get("tasks") or []

    absolute_types = {
        "hydration",
        "eye",
        "stretch",
        "walk"
    }

    for task in tasks:

        if task.get("type") not in absolute_types:
            continue

        task["times"] = [
            f"{h:02d}:{m:02d}"
            for h, m in explicit_times
        ]

        task.pop("interval_minutes", None)
        task.pop("start_time", None)
        task.pop("end_time", None)

    if not parsed.get("active_window"):
        parsed["active_window"] = {
            "start": "00:00",
            "end": "23:59"
        }

    return parsed


def force_window_from_text(parsed, user_text):
    """
    If the user's message contains an explicit 'from X to Y' phrase,
    hydration/eye/stretch/walk tasks must be stored as a start_time/
    end_time window, never as an absolute times list. This prevents
    e.g. "from 10am to 10:15am" from being mis-parsed as two separate
    alarm points instead of a single 15-minute window.
    """

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
    """
    Detect a duration the user explicitly stated in words, e.g.
    "30 mins stretch break". Returns None if no duration is stated,
    or if the number belongs to an "every N minutes" interval phrase
    (which is not a duration claim).
    """

    if not isinstance(text, str):
        return None

    match = re.search(
        r"\b(\d+)\s*(?:minutes?|mins?)\b",
        text.lower()
    )

    if not match:
        return None

    prefix = text.lower()[:match.start()].rstrip()

    if prefix.endswith("every"):
        return None

    return int(match.group(1))


def extract_stated_snooze_minutes(text):
    """
    Detect an explicitly stated snooze/reminder-period duration,
    e.g. "30 mins snooze period", "snooze of 20 minutes",
    "15 min reminder period".
    """

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


def apply_medication_extras(plan, user_text):
    """
    Fill in per-medicine snooze_min (from stated text, if any) and
    gap_min (computed from the actual dose spacing, which is far
    more reliable than trying to regex the user's gap phrasing).
    """

    stated_snooze = extract_stated_snooze_minutes(user_text)

    for med in plan.get("medication", []) or []:

        doses = med.get("doses", [])

        if len(doses) >= 2:

            first = doses[0]["h"] * 60 + doses[0]["m"]
            second = doses[1]["h"] * 60 + doses[1]["m"]

            gap = second - first

            if gap < 0:
                gap += 1440

            med["gap_min"] = gap

        else:

            med["gap_min"] = 0

        if stated_snooze is not None:

            med["snooze_min"] = stated_snooze

    return plan


def extract_medication_window(text):
    """
    Detect an explicit 'from X to Y' time window in the raw user
    text, tolerant of missing whitespace around 'to' (e.g. the
    common typo "8amto 8:30am").
    """

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
    """
    Detect an explicitly stated gap between doses, e.g.
    "45 mins gap" or "gap of 45 minutes".
    """

    if not isinstance(text, str):
        return None

    match = re.search(
        r"(\d+)\s*(?:minutes?|mins?)\s*gap",
        text.lower()
    )

    if match:
        return int(match.group(1))

    match = re.search(
        r"gap\s*(?:of\s*)?(\d+)\s*(?:minutes?|mins?)",
        text.lower()
    )

    if match:
        return int(match.group(1))

    return None


def check_medication_mismatch(plan, user_text):
    """
    If the user states both a time window ("from 8am to 8:30am")
    and a gap between doses ("45 mins gap") that don't fit together
    -- more doses than the window can actually hold at that
    spacing -- flag it as invalid instead of silently dropping or
    mis-scheduling it.
    """

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
                    f"medication '"
                    f"{med.get('label', 'Medication')}': "
                    f"{count} doses with a {gap} min gap need "
                    f"{needed} min, but the stated window "
                    f"{sh:02d}:{sm:02d}"
                    f"→"
                    f"{eh:02d}:{em:02d} "
                    f"is only {span} min"
                )
            })

            med["_invalid"] = True

    return mismatches


def check_duration_mismatch(plan, user_text):
    """
    Compare a stated duration (e.g. "30 mins") against the actual
    start_time/end_time window on each task. Returns a list of
    invalid-input entries for any task where they disagree.
    """

    stated = extract_stated_duration_minutes(user_text)

    if stated is None:
        return []

    mismatches = []

    for t in plan.get("tasks", []):

        start = t.get("start_time")
        end = t.get("end_time")

        if not start or not end:
            continue

        span = window_duration_minutes(
            start[0], start[1],
            end[0], end[1]
        )

        if span and span != stated:

            mismatches.append({
                "task": t.get("type"),
                "reason": (
                    f"{t.get('type')}: stated duration {stated} min "
                    f"does not match window "
                    f"{start[0]:02d}:{start[1]:02d}"
                    f"→"
                    f"{end[0]:02d}:{end[1]:02d} "
                    f"({span} min)"
                )
            })

    return mismatches


def apply_days_from_user_text(parsed, user_text):
    """
    Safety net for explicit day selections.
    """

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

    ABSOLUTE_TASK_TYPES = {
        "hydration",
        "eye",
        "stretch",
        "walk"
    }

    for t in tasks:

        task_type = t.get("type")

        times = t.get("times") or []

        parsed_times = []

        for ts in times:

            pt = parse_time(ts)

            if pt:
                parsed_times.append({
                    "h": pt[0],
                    "m": pt[1]
                })

        start_time = parse_time(
            t.get("start_time")
        )

        end_time = parse_time(
            t.get("end_time")
        )

        interval_minutes = safe_int(
            t.get("interval_minutes")
        )

        # SAFETY NET 1
        if (
            task_type in ABSOLUTE_TASK_TYPES
            and not parsed_times
            and start_time
            and not end_time
            and interval_minutes is None
        ):

            parsed_times = [{
                "h": start_time[0],
                "m": start_time[1]
            }]

            start_time = None

        # SAFETY NET 2
        if (
            task_type in ABSOLUTE_TASK_TYPES
            and not parsed_times
            and start_time
            and end_time
            and start_time == end_time
            and interval_minutes is None
        ):

            parsed_times = [{
                "h": start_time[0],
                "m": start_time[1]
            }]

            start_time = None
            end_time = None

        mode = (
            "absolute"
            if parsed_times
            else "interval"
        )

        out.append({
            "type": task_type,
            "mode": mode,
            "interval_minutes": interval_minutes,
            "times": parsed_times,
            "start_time": start_time,
            "end_time": end_time,
            "focus_min": safe_int(
                t.get("focus_min")
            ),
            "break_min": safe_int(
                t.get("break_min")
            ),
            "cycles": safe_int(
                t.get("cycles")
            ),
            "days": normalize_days(
                t.get("days")
            )
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
                doses.append({
                    "h": pt[0],
                    "m": pt[1]
                })

        if not doses:
            continue

        start, end = resolve_dates(
            m.get("start"),
            m.get("end")
        )

        out.append({
            "label": m.get(
                "label",
                "Medication"
            ),
            "start": start,
            "end": end,
            "days": normalize_days(
                m.get("days")
            ),
            "doses": doses
        })

    return out


def normalize_custom(parsed):
    """
    Custom (freeform) reminders — anything that isn't hydration,
    eye, stretch, walk, meditation, pomodoro, or medication.
    Each entry is either one-time ("date" set) or recurring
    ("days" set).
    """

    items = parsed.get("custom") or []

    out = []

    for c in items:

        times = []

        for t in (c.get("times") or []):

            pt = parse_time(t)

            if pt:
                times.append(
                    f"{pt[0]:02d}:{pt[1]:02d}"
                )

        if not times:
            continue

        entry = {
            "label": c.get(
                "label",
                "Reminder"
            ),
            "times": times
        }

        date_val = c.get("date")

        if date_val:
            entry["date"] = date_val
            entry["repeat"] = False

        else:
            entry["days"] = normalize_days(
                c.get("days")
            )
            entry["repeat"] = True

        out.append(entry)

    return out


def build_plan(parsed):

    active = parsed.get(
        "active_window"
    ) or {}

    return {
        "tasks": normalize_tasks(parsed),

        "medication": normalize_medication(
            parsed
        ),

        "custom": normalize_custom(
            parsed
        ),

        "dnd": parsed.get(
            "do_not_disturb",
            []
        ),

        "hydration_goal_ml":
            parsed.get(
                "_hydration_goal_ml"
            ),

        "active_window": {
            "start": active.get(
                "start",
                "00:00"
            ),
            "end": active.get(
                "end",
                "23:59"
            )
        }
    }


# =========================================================
# OLD CONVERTER
# =========================================================

def convert_to_device_schema(plan):

    config = json.loads(
        json.dumps(BASE_CONFIG)
    )

    config["_meta"]["ts_written"] = int(
        datetime.now(IST).timestamp()
    )

    active = plan.get(
        "active_window"
    ) or {
        "start": "00:00",
        "end": "23:59"
    }

    global_start = parse_time(
        active.get("start") or "00:00"
    )

    global_end = parse_time(
        active.get("end") or "23:59"
    )

    for t in plan.get("tasks", []):

        start = (
            t.get("start_time")
            or global_start
        )

        end = (
            t.get("end_time")
            or global_end
        )

        sh, sm = (
            start if start else (0, 0)
        )

        eh, em = (
            end if end else (23, 59)
        )

        def apply(
            task_key,
            interval_key,
            default_interval
        ):

            days = normalize_days(
                t.get("days")
            )

            if (
                t.get("mode") == "absolute"
                and t.get("times")
            ):

                config[task_key].update({
                    "enabled": True,
                    "mode": "absolute",
                    "days": days,
                    "abs": {
                        "enabled": True,
                        "times": t["times"]
                    }
                })

            else:

                config[task_key].update({
                    "enabled": True,
                    "mode": "interval",
                    "days": days,
                    interval_key:
                        (
                            t.get(
                                "interval_minutes"
                            )
                            or default_interval
                        )
                        * (
                            60000
                            if "ms" in interval_key
                            else 1
                        ),
                    "start_hour": sh,
                    "start_min": sm,
                    "end_hour": eh,
                    "end_min": em
                })

        if t["type"] == "hydration":

            apply(
                "hydration",
                "interval_ms",
                30
            )

        elif t["type"] == "eye":

            apply(
                "eye",
                "interval_ms",
                20
            )

        elif t["type"] == "stretch":

            apply(
                "stretch",
                "interval_ms",
                60
            )

            if t.get("mode") != "absolute":

                config["stretch"]["phases"] = [{
                    "sh": sh,
                    "sm": sm,
                    "eh": eh,
                    "em": em
                }]

        elif t["type"] == "walk":

            apply(
                "walk",
                "interval_min",
                120
            )

        elif t["type"] == "meditation":

            if start and end:

                sh, sm = start
                eh, em = end

                duration_sec = (
                    (
                        (eh * 60 + em)
                        - (sh * 60 + sm)
                    )
                    * 60
                )

                config["meditation"].update({
                    "enabled": True,
                    "sh": sh,
                    "sm": sm,
                    "eh": eh,
                    "em": em,
                    "display_sec":
                        max(
                            duration_sec,
                            60
                        ),
                    "days":
                        normalize_days(
                            t.get("days")
                        )
                })

        elif t["type"] == "pomodoro":

            config["pomo"]["enabled"] = True

            config["pomo"]["focus_min"] = (
                t.get("focus_min") or 25
            )

            config["pomo"]["break_min"] = (
                t.get("break_min") or 5
            )

            config["pomo"]["cycles"] = (
                t.get("cycles") or 4
            )

            if (
                not t.get("start_time")
                and not t.get("end_time")
            ):

                config["pomo"][
                    "lap_mode_enabled"
                ] = True

                if global_start and global_end:

                    gsh, gsm = global_start
                    geh, gem = global_end

                    config["pomo"]["laps"] = [{
                        "sh": gsh,
                        "sm": gsm,
                        "eh": geh,
                        "em": gem,
                        "enabled": True
                    }]

            else:

                config["pomo"][
                    "lap_mode_enabled"
                ] = False

                config["pomo"]["laps"] = []

    if plan.get("medication"):

        config["medication_cfg"][
            "enabled"
        ] = True

        config["medication"] = plan[
            "medication"
        ]

    return config


# =========================================================
# NEW V6 BASE CONFIG
# =========================================================

BASE_CONFIG_V2 = {

    "_meta": {
        "schema_ver": 6,
        "device": "FROST"
    },

    "reminders": {

        "hydration": {
            "enabled": False,
            "mode": "interval",
            "interval_ms": 3600000,
            "display_ms": 60000,
            "require_ack": True,
            "goal_ml": 0,
            "start_hour": 8,
            "start_min": 0,
            "end_hour": 20,
            "end_min": 0,
            "days": ALL_DAYS[:],
            "abs": {
                "times": []
            }
        },

        "stretch": {
            "enabled": False,
            "mode": "interval",
            "interval_ms": 3600000,
            "display_ms": 60000,
            "require_ack": True,
            "start_hour": 8,
            "start_min": 0,
            "end_hour": 20,
            "end_min": 0,
            "days": ALL_DAYS[:],
            "abs": {
                "times": []
            }
        },

        "eye": {
            "enabled": False,
            "mode": "interval",
            "interval_ms": 1800000,
            "display_ms": 60000,
            "require_ack": True,
            "start_hour": 8,
            "start_min": 0,
            "end_hour": 20,
            "end_min": 0,
            "days": ALL_DAYS[:],
            "abs": {
                "times": []
            }
        },

        "walk": {
            "enabled": False,
            "mode": "interval",
            "interval_ms": 7200000,
            "display_ms": 60000,
            "require_ack": True,
            "start_hour": 8,
            "start_min": 0,
            "end_hour": 20,
            "end_min": 0,
            "days": ALL_DAYS[:],
            "abs": {
                "times": []
            }
        },

        "meditation": {
            "enabled": False,
            "sh": 0,
            "sm": 0,
            "eh": 0,
            "em": 0,
            "display_sec": 600,
            "require_ack": True,
            "days": ALL_DAYS[:]
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
            "enabled": False,
            "require_dock": True,
            "tracks": [45]
        },
        "healing_schedules": []
    },

    "bottle_clean": {
        "enabled": False,
        "interval_days": 1,
        "hour": 18,
        "minute": 0,
        "display_ms": 60000,
        "require_ack": True
    },

    "pomodoro": {
        "enabled": False,
        "focus_min": 25,
        "break_min": 5,
        "cycles": 4,
        "auto_start_break": True,
        "auto_start_focus": True,
        "lap_mode_enabled": True,
        "laps": [],
        "focus_counter": {
            "x": 102,
            "y": 125,
            "text_size": 1,
            "text_color": 0,
            "text_align": 1
        },
        "break_counter": {
            "x": 120,
            "y": 150,
            "text_size": 1,
            "text_color": 0,
            "text_align": 1
        }
    }
}


# =========================================================
# NEW V6 CONVERTER
# =========================================================

def convert_to_new_schema(plan):

    config = json.loads(
        json.dumps(BASE_CONFIG_V2)
    )

    active = plan.get(
        "active_window"
    ) or {
        "start": "00:00",
        "end": "23:59"
    }

    global_start = parse_time(
        active.get("start") or "00:00"
    )

    global_end = parse_time(
        active.get("end") or "23:59"
    )

    reminders = config["reminders"]

    invalid = []

    # Hydration goal

    if plan.get("hydration_goal_ml") is not None:

        reminders["hydration"][
            "goal_ml"
        ] = plan[
            "hydration_goal_ml"
        ]

    for t in plan.get("tasks", []):

        start = (
            t.get("start_time")
            or global_start
        )

        end = (
            t.get("end_time")
            or global_end
        )

        sh, sm = (
            start if start else (0, 0)
        )

        eh, em = (
            end if end else (23, 59)
        )

        def apply(
            task_key,
            default_interval_min
        ):

            days = normalize_days(
                t.get("days")
            )

            if (
                t.get("mode") == "absolute"
                and t.get("times")
            ):

                reminders[task_key].update({
                    "enabled": True,
                    "mode": "absolute",
                    "days": days,
                    "abs": {
                        "times": t["times"]
                    }
                })

            else:

                span = window_duration_minutes(
                    sh,
                    sm,
                    eh,
                    em
                )

                interval_min = (
                    t.get("interval_minutes")
                    or default_interval_min
                )

                if span is None or span <= 0:

                    invalid.append({
                        "task": task_key,
                        "reason":
                            f"{task_key} window "
                            f"{sh:02d}:{sm:02d}"
                            f"→"
                            f"{eh:02d}:{em:02d} "
                            f"is zero-length or reversed"
                    })

                    return

                if interval_min > span:

                    invalid.append({
                        "task": task_key,
                        "reason":
                            f"{task_key}: every "
                            f"{interval_min} min "
                            f"doesn't fit in the "
                            f"{span}-min window "
                            f"{sh:02d}:{sm:02d}"
                            f"→"
                            f"{eh:02d}:{em:02d}"
                    })

                    return

                reminders[task_key].update({
                    "enabled": True,
                    "mode": "interval",
                    "days": days,
                    "interval_ms":
                        interval_min * 60000,
                    "start_hour": sh,
                    "start_min": sm,
                    "end_hour": eh,
                    "end_min": em
                })

        # HYDRATION

        if t["type"] == "hydration":

            apply(
                "hydration",
                30
            )

        # EYE

        elif t["type"] == "eye":

            apply(
                "eye",
                20
            )

        # STRETCH

        elif t["type"] == "stretch":

            apply(
                "stretch",
                60
            )

        # WALK

        elif t["type"] == "walk":

            apply(
                "walk",
                120
            )

        # MEDITATION

        elif t["type"] == "meditation":

            if start and end:

                sh, sm = start
                eh, em = end

                span = window_duration_minutes(
                    sh,
                    sm,
                    eh,
                    em
                )

                if span is None or span <= 0:

                    invalid.append({
                        "task": "meditation",
                        "reason":
                            f"meditation window "
                            f"{sh:02d}:{sm:02d}"
                            f"→"
                            f"{eh:02d}:{em:02d} "
                            f"is zero-length or reversed"
                    })

                else:

                    reminders[
                        "meditation"
                    ].update({
                        "enabled": True,
                        "sh": sh,
                        "sm": sm,
                        "eh": eh,
                        "em": em,
                        "display_sec":
                            span * 60,
                        "days":
                            normalize_days(
                                t.get("days")
                            )
                    })

        # POMODORO

        elif t["type"] == "pomodoro":

            pomo = config["pomodoro"]

            pomo["enabled"] = True

            pomo["focus_min"] = (
                t.get("focus_min")
                or 25
            )

            pomo["break_min"] = (
                t.get("break_min")
                or 5
            )

            pomo["cycles"] = (
                t.get("cycles")
                or 4
            )

            # No explicit timing

            if (
                not t.get("start_time")
                and not t.get("end_time")
            ):

                pomo[
                    "lap_mode_enabled"
                ] = True

                if global_start and global_end:

                    gsh, gsm = global_start
                    geh, gem = global_end

                    pomo["laps"] = [{
                        "sh": gsh,
                        "sm": gsm,
                        "eh": geh,
                        "em": gem,
                        "enabled": True
                    }]

            # Explicit timing

            else:

                span = window_duration_minutes(
                    sh,
                    sm,
                    eh,
                    em
                )

                cycles = pomo["cycles"]

                needed = (
                    pomo["focus_min"] * cycles
                    +
                    pomo["break_min"]
                    * max(cycles - 1, 0)
                )

                if span is None or span <= 0:

                    invalid.append({
                        "task": "pomodoro",
                        "reason":
                            f"pomodoro window "
                            f"{sh:02d}:{sm:02d}"
                            f"→"
                            f"{eh:02d}:{em:02d} "
                            f"is zero-length or reversed"
                    })

                    pomo["enabled"] = False

                elif needed > span:

                    invalid.append({
                        "task": "pomodoro",
                        "reason":
                            f"pomodoro: {cycles} cycles "
                            f"of "
                            f"{pomo['focus_min']}/"
                            f"{pomo['break_min']} min "
                            f"need {needed} min but "
                            f"the window "
                            f"{sh:02d}:{sm:02d}"
                            f"→"
                            f"{eh:02d}:{em:02d} "
                            f"is only {span} min"
                    })

                    pomo["enabled"] = False

                else:

                    pomo[
                        "lap_mode_enabled"
                    ] = False

                    pomo["laps"] = []

    # =====================================================
    # MEDICATION
    # =====================================================

    if plan.get("medication"):

        valid_meds = [
            m for m in plan["medication"]
            if not m.get("_invalid")
        ]

        if valid_meds:

            reminders[
                "medication"
            ]["enabled"] = True

            medicines = []

            for m in valid_meds:

                medicines.append({

                    "id":
                        f"med_{len(medicines) + 1:03d}",

                    "label":
                        m.get(
                            "label",
                            "Medication"
                        ),

                    "enabled": True,

                    "start":
                        m.get("start"),

                    "end":
                        m.get("end"),

                    "days":
                        m.get("days"),

                    "text_x": 120,
                    "text_y": 135,
                    "text_size": 1,
                    "text_color": 65535,
                    "text_align": 1,
                    "text_width": 180,

                    "doses":
                        m.get("doses", []),

                    "snooze_min":
                        m.get(
                            "snooze_min",
                            reminders["medication"].get(
                                "snooze_min", 15
                            )
                        ),

                    "gap_min":
                        m.get("gap_min", 0)
                })

            # Keep the global medication snooze in sync with
            # whatever the user actually stated, so the summary
            # line above the per-medicine cards matches too.

            stated_snooze_values = [
                m.get("snooze_min")
                for m in valid_meds
                if m.get("snooze_min") is not None
            ]

            if stated_snooze_values:

                reminders[
                    "medication"
                ]["snooze_min"] = stated_snooze_values[0]

            reminders[
                "medication"
            ]["medicines"] = medicines

    # =====================================================
    # CUSTOM REMINDERS
    # =====================================================

    if plan.get("custom"):

        events = []

        for c in plan["custom"]:

            repeat = c.get("repeat", True)

            events.append({
                "id":
                    f"custom_{len(events) + 1:03d}",

                "label":
                    c.get("label", "Reminder"),

                "enabled": True,

                "repeat": repeat,

                "date":
                    c.get("date") if not repeat else None,

                "days":
                    c.get("days", ALL_DAYS[:]) if repeat else [],

                "times":
                    c.get("times", [])
            })

        if events:

            reminders[
                "custom"
            ]["enabled"] = True

            reminders[
                "custom"
            ]["events"] = events

    return config, invalid


# =========================================================
# ROUTE
# =========================================================

@app.route("/parse", methods=["POST"])
def parse_schedule():

    logs = []

    start_time = time.perf_counter()

    try:

        data = request.get_json()

        logs.append(
            "Step 1: Input received"
        )

        today_ist = datetime.now(IST)

        dated_system_prompt = (
            SYSTEM_PROMPT
            + "\n\nTODAY'S DATE: "
            + today_ist.strftime("%Y-%m-%d (%A)")
        )

        response = client.responses.create(

            model="gpt-5-nano",

            input=[

                {
                    "role": "system",
                    "content": dated_system_prompt
                },

                {
                    "role": "user",
                    "content": data["text"]
                }
            ]
        )

        raw = response.output_text

        logs.append(
            "Step 2: LLM response received"
        )

        logs.append(
            f"Preview: {raw[:120]}..."
        )

        parsed = safe_json_parse(
            raw
        )

        # Casual conversation short-circuit — greetings, thanks,
        # "what is frost", etc. Skip all schedule-building below.

        if isinstance(parsed, dict) and "chat_reply" in parsed:

            logs.append(
                "Step 2b: Casual conversation detected, "
                "skipping schedule parsing"
            )

            elapsed = (
                time.perf_counter()
                - start_time
            ) * 1000

            logs.append(
                f"⏱ Total time: "
                f"{round(elapsed, 2)} ms"
            )

            return jsonify({
                "reply": parsed["chat_reply"],
                "logs": logs
            })

        # Force exact "at <time>"

        parsed = force_absolute_times_from_user_text(
            parsed,
            data.get("text", "")
        )

        time_parse_invalid = parsed.pop("_invalid_times", [])

        # Force "from X to Y" phrasing into a real window
        parsed = force_window_from_text(
            parsed,
            data.get("text", "")
        )

        # Apply hydration goal

        parsed = apply_water_goal(
            parsed,
            data.get("text", "")
        )

        # Apply day rules

        parsed = apply_days_from_user_text(
            parsed,
            data.get("text", "")
        )

        logs.append(
            "Step 3: JSON parsed, "
            "exact-time, hydration-goal "
            "and day rules applied"
        )

        plan = build_plan(parsed)

        plan = apply_medication_extras(
            plan,
            data.get("text", "")
        )

        # Final defensive check

        if (
            plan.get(
                "active_window",
                {}
            ).get("start")
            ==
            plan.get(
                "active_window",
                {}
            ).get("end")
        ):

            plan["active_window"] = {
                "start": "00:00",
                "end": "23:59"
            }

        logs.append(
            f"Step 4: Tasks="
            f"{len(plan['tasks'])}, "
            f"Meds="
            f"{len(plan['medication'])}"
        )

        # Stated-duration vs actual-window validation

        duration_mismatches = check_duration_mismatch(
            plan,
            data.get("text", "")
        )

        medication_mismatches = check_medication_mismatch(
            plan,
            data.get("text", "")
        )

        # NEW V6 schema

        config, invalid = convert_to_new_schema(
            plan
        )

        invalid = (
            time_parse_invalid
            + duration_mismatches
            + medication_mismatches
            + invalid
        )

        for item in duration_mismatches:

            task_cfg = config.get(
                "reminders", {}
            ).get(item["task"])

            if task_cfg is not None:
                task_cfg["enabled"] = False

        logs.append(
            "Step 5: Device config generated "
            "(v6 schema)"
        )

        if invalid:

            for item in invalid:

                logs.append(
                    f"⚠ Rejected "
                    f"(invalid input): "
                    f"{item['reason']}"
                )

        reminders_cfg = config.get(
            "reminders",
            {}
        )

        enabled = [
            k
            for k in [
                "hydration",
                "eye",
                "stretch",
                "walk",
                "meditation",
                "custom"
            ]
            if reminders_cfg.get(
                k,
                {}
            ).get("enabled")
        ]

        if config.get(
            "pomodoro",
            {}
        ).get("enabled"):

            enabled.append(
                "pomodoro"
            )

        logs.append(
            "Enabled: "
            +
            (
                ", ".join(enabled)
                if enabled
                else "none"
            )
        )

        medicines = (
            reminders_cfg
            .get(
                "medication",
                {}
            )
            .get(
                "medicines",
                []
            )
        )

        if medicines:

            logs.append(
                "Medication count: "
                f"{len(medicines)}"
            )

        custom_events = (
            reminders_cfg
            .get(
                "custom",
                {}
            )
            .get(
                "events",
                []
            )
        )

        if custom_events:

            logs.append(
                "Custom reminder count: "
                f"{len(custom_events)}"
            )

        elapsed = (
            time.perf_counter()
            - start_time
        ) * 1000

        logs.append(
            f"⏱ Total time: "
            f"{round(elapsed, 2)} ms"
        )

        return jsonify({

            "data": config,

            "active_window":
                plan.get(
                    "active_window"
                ),

            "invalid":
                invalid,

            "logs":
                logs
        })

    except Exception as e:

        elapsed = (
            time.perf_counter()
            - start_time
        ) * 1000

        logs.append(
            f"ERROR: {str(e)}"
        )

        logs.append(
            f"⏱ Total time: "
            f"{round(elapsed, 2)} ms"
        )

        return jsonify({

            "error": str(e),

            "logs": logs

        }), 400


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

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )