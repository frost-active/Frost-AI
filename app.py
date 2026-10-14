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

"hydration" is ONLY for the user DRINKING water. Watering plants,
flowers or a garden ("water the plants") is a custom habit, not
hydration. "Change hydration from 10am to 11am" edits the hydration
reminder; it never creates a medication.

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

- A short length in front of the task ("5 minute desk stretch every
  3 hours from 9am to 6pm") is the length of one break. The repeat gap
  is the "every ..." part (180 minutes). Windows may run past midnight
  ("11pm to 3am").

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

- If the user gives a start time and a LENGTH (e.g. "meditate at
  7am for 20 minutes"), set end_time = start_time + length.

- If the user gives neither an end time nor a length, OMIT
  end_time (the server will ask).


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

- "start_time" is when the pomodoro session starts. "end_time" is
  optional.

- If cycles are not stated, OMIT "cycles" (the server works it out).

- If the user gave no start time at all, OMIT start_time (the
  server will ask the user for it).

- Do NOT create laps and do NOT use lap_mode_enabled.


BOTTLE CLEANING RULES:

- "clean my bottle at 6pm every day" ->

{
  "type": "bottle_clean",
  "times": ["18:00"],
  "interval_days": 1
}

- "interval_days" is the number of days BETWEEN cleanings
  (default 1 = every day). Only ONE time of day is used.

- "every 2 days" is an interval (interval_days = 2), NOT a
  duration.


MEDICATION FORMAT:

{
  "label": "string",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "duration_days": int,
  "days": ["mon","tue","wed","thu","fri","sat","sun"],
  "times": ["HH:MM","HH:MM"]
}

- start_date / end_date / duration_days are optional and follow
  the DATE RANGE RULES below.

- "days" follows the same DAY RULES as tasks above:
  only the days the user actually named, or omit for every day.


NEVER INVENT TIMES:

- If the user did not state any time for a reminder / habit /
  medication (no "at ...", no "every X minutes", no "from X to Y"),
  leave "times", "interval_minutes", "start_time" and "end_time"
  OUT. Never guess a default time. The server asks the user.

- Still return the item itself (type or label + dates), so the
  server knows what the user wants.


DATE RANGE RULES (every task, medication entry and custom habit):

- Every reminder has a start date and an end date. Use TODAY'S DATE
  (given at the bottom) to resolve anything relative.

- If the user states a length of time ("for 10 days", "for 2 weeks",
  "for a month", "eye break of 10 days", "next 5 days"):
  set "duration_days" to the number of CALENDAR days
  (1 week = 7), or for months set "duration_months" to the number of
  months. "for 8 weeks" is duration_days 56 no matter which weekdays
  it runs on - never count sessions. Do NOT work out the dates
  yourself; the server counts from today.

- If the user states real dates ("from 12 oct to 20 oct",
  "until 25 oct", "starting monday"): set "start_date" and/or
  "end_date" as YYYY-MM-DD.

- If the user wants to lengthen an existing reminder ("extend eye
  break by 5 days", "add 3 more days"): set "extend_days".

- If no duration or dates are mentioned: OMIT all of these fields
  (the server applies a default and tells the user).

- A duration is NOT a time of day, and "every 2 days" is NOT a
  duration. "for 15 minutes" is the LENGTH of one session: never turn
  it into a second reminder time.

- "daily" / "every day" with no duration or dates: OMIT every date
  field (the reminder is open-ended).

Example:

User: "eye break at 10:15am and 2pm for 10 days"

{
  "tasks": [
    {
      "type": "eye",
      "times": ["10:15", "14:00"],
      "duration_days": 10
    }
  ]
}

Example (no time given -> leave times out, the server will ask):

User: "i want eye break for 10 days"

{
  "tasks": [
    {
      "type": "eye",
      "duration_days": 10
    }
  ]
}


EXCLUDED DAYS:

- If the user says NOT to remind on some days ("not on sat and sun",
  "except sunday", "skip saturdays", "no reminders on sun"), set
  "days" to ALL seven days MINUS the excluded ones, e.g.
  "not on sat sun" -> ["mon","tue","wed","thu","fri"].

- Still give only "duration_days" (e.g. 10). The server counts the
  10 days using ONLY the remaining days, so excluded days do not
  use up the count.

Example:

User: "eye break at 10:15am and 2pm for 10 days, not on sat and sun"

{
  "tasks": [
    {
      "type": "eye",
      "times": ["10:15", "14:00"],
      "days": ["mon","tue","wed","thu","fri"],
      "duration_days": 10
    }
  ]
}


ACTION RULES (changing what already exists):

- Every task, medication entry and custom habit may carry:

    "action": "set"      (default: create it, or update it if it
                          already exists)
    "action": "disable"  (turn off / stop / pause / cancel it)
    "action": "remove"   (delete it completely)

- When the user CHANGES something that already exists
  ("change eye break to 5pm", "make chess 7pm", "move walk to
  weekdays", "extend chess by 5 days"), return ONLY the type (or
  label) plus the fields that change. Do not repeat unchanged
  fields.

- If the user says "also", "add", or "one more time" for an
  existing reminder (e.g. "also remind me at 5pm"), set
  "merge_times": true so the new times are added to the old ones.

- To rename a custom habit, keep "label" as the OLD name and put
  the new name in "new_label".

Example:

User: "stop the walk reminders"

{
  "tasks": [
    { "type": "walk", "action": "disable" }
  ]
}

Example:

User: "delete the chess habit"

{
  "custom": [
    { "label": "Play chess", "action": "remove" }
  ]
}


CUSTOM HABIT / REMINDER RULES:

- A HABIT is anything the user wants to do on a schedule that is
  NOT hydration, eye, stretch, walk, meditation, pomodoro, bottle
  cleaning, or medication - e.g. "play chess at 6pm for 10 days",
  "practice guitar every weekday at 7pm", "read a book at 10pm",
  "remind me to call mom", "submit the assignment".

- Put each one in the top-level "custom" array (NOT in "tasks").

- "label" is a short activity name in plain words
  (e.g. "Play chess", "Read a book").

- Recurring habit / reminder:

{
  "label": "Play chess",
  "days": ["mon","wed","fri"],
  "times": ["18:00"],
  "duration_days": 10
}

  - Follow the same DAY RULES as tasks. If the user says
    "every day"/"daily" or names no day, OMIT "days".
  - Follow the DATE RANGE RULES above for the length of the habit.

- ONE-TIME reminder - a specific calendar date, "today",
  "tomorrow", a single named upcoming weekday meant as one
  occurrence, or the word "once"/"one time":

{
  "label": "Call mom",
  "date": "YYYY-MM-DD",
  "times": ["18:00"]
}

  - Resolve relative dates ("tomorrow", "next friday") to a real
    YYYY-MM-DD date using TODAY'S DATE.
  - Do NOT include "days" or duration fields for a one-time reminder.

- A single "at" can introduce a list of times, same as ABSOLUTE
  MODE above (e.g. "play chess at 11am and 3pm").

Example:

User: "remind me on 10am every monday to submit assignment"

{
  "custom": [
    {
      "label": "Submit assignment",
      "days": ["mon"],
      "times": ["10:00"]
    }
  ]
}

Example:

User: "i want to play chess at 6pm for 15 days"

{
  "custom": [
    {
      "label": "Play chess",
      "times": ["18:00"],
      "duration_days": 15
    }
  ]
}

Example:

User: "change chess to 7pm"

{
  "custom": [
    {
      "label": "Play chess",
      "times": ["19:00"]
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
# HELPERS
# =========================================================

# How long a reminder / habit runs (in calendar days) when the user
# gives no duration and no dates. The chatbot always tells the user when this default
# is used.
DEFAULT_DURATION_DAYS = 7

# "for 10 days" -> end_date = today + 10 days (False), or
# today + 9 days so that today counts as day 1 (True).
DURATION_COUNTS_TODAY = False

# "daily" with no duration is open-ended. The device schema still needs
# an end_date, so an ongoing reminder is given this many days.
ONGOING_DAYS = 365

TASK_TYPES = {
    "hydration",
    "eye",
    "stretch",
    "walk",
    "meditation",
    "pomodoro",
    "bottle_clean"
}

# Task types that can be a list of exact clock times
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
    Minutes from a start to an end clock time. If the end is earlier
    than the start the window runs past midnight (23:00 -> 03:00 is
    240 min). Zero means start and end are the same (empty window).
    """
    try:
        diff = (
            int(eh) * 60
            + int(em)
            - int(sh) * 60
            - int(sm)
        )
    except (TypeError, ValueError):
        return None

    if diff < 0:
        diff += 24 * 60

    return diff


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


def add_months(d, months):
    """Calendar month arithmetic (12 Oct + 2 months = 12 Dec)."""

    idx = d.year * 12 + (d.month - 1) + months
    year, month = divmod(idx, 12)
    month += 1

    last = (
        date(year + (month == 12), month % 12 + 1, 1)
        - timedelta(days=1)
    ).day

    return date(year, month, min(d.day, last))


def span_end(start, n=0, months=0):
    """
    End date for "N days / N months from start". Durations are plain
    calendar time: "8 weeks" is 56 days and "2 months" is two calendar
    months, no matter which weekdays the reminder runs on.
    """

    end = add_months(start, months) if months else start

    end += timedelta(days=n or 0)

    if DURATION_COUNTS_TODAY:
        end -= timedelta(days=1)

    return end


def norm_days_opt(days):
    """Like normalize_days, but None when the user gave no days."""

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

    return s[0].upper() + s[1:]


def parse_time_list(values):
    """['10:15','14:00'] -> [{'h':10,'m':15},{'h':14,'m':0}] (deduped)."""

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


# ---------------------------------------------------------
# Display helpers (used for the confirmation summary)
# ---------------------------------------------------------

def fmt_time12(h, m):
    suffix = "AM" if h < 12 else "PM"
    return f"{(h % 12) or 12}:{m:02d} {suffix}"


def fmt_times(times):
    return ", ".join(
        fmt_time12(t["h"], t["m"]) for t in times
    )


def fmt_date(value):
    d = parse_iso_date(value)
    return d.strftime("%d %b %Y") if d else str(value)


def fmt_when(days, start_d, end_d, ongoing=False):
    """
    Days + date range for the confirmation text. An open-ended daily
    reminder just says "Daily" instead of showing a start/end range.
    """

    if ongoing:

        base = (
            "Daily"
            if normalize_days(days) == ALL_DAYS
            else fmt_days(days)
        )

        if start_d > today_ist():
            base += f" from {fmt_date(start_d)}"

        return base

    if start_d == end_d:
        return f"{start_d.strftime('%a')} {fmt_date(start_d)} (one time)"

    return (
        f"{fmt_days(days)} · "
        f"{fmt_date(start_d)} → {fmt_date(end_d)}"
    )


def fmt_days(days):
    days = normalize_days(days)

    if days == ALL_DAYS:
        return "every day"

    if days == WEEKDAYS:
        return "weekdays"

    if days == WEEKEND:
        return "weekends"

    return ", ".join(d.capitalize() for d in days)


# ---------------------------------------------------------
# Duration / date-range parsing
# ---------------------------------------------------------

_NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20, "thirty": 30
}

# "for 10 days", "of 2 weeks", "next 3 months", "10 days",
# "10-day". "every 2 days" and "twice a day" are NOT durations.
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
    """
    Every stated duration in the text as (count, unit) where unit is
    "days" or "months". Weeks are folded into days (1 week = 7 days);
    months stay months so they can be added as calendar months.
    """

    out = []

    if not isinstance(text, str):
        return out

    for m in _DURATION_RE.finditer(text):

        raw = (m.group("n1") or m.group("n2") or "").lower()

        n = int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw)

        if not n:
            continue

        unit = m.group("unit").lower()

        if unit.startswith("month"):
            out.append((n, "months"))
        elif unit.startswith("week"):
            out.append((n * 7, "days"))
        else:
            out.append((n, "days"))

    return out


_DAY_NAMES_FULL = {
    "mon": "monday", "tue": "tuesday", "wed": "wednesday",
    "thu": "thursday", "fri": "friday", "sat": "saturday",
    "sun": "sunday"
}

_DAY_RANGE_RE = re.compile(
    r"\b(" + _WEEKDAY_NAMES + r")\s*(?:through|thru|to|until|till|"
    r"-|\u2013|\u2014)\s*(" + _WEEKDAY_NAMES + r")\b",
    re.IGNORECASE
)


def expand_day_ranges(text):
    """
    "monday through friday" / "mon-fri" / "sat to mon" ->
    "monday, tuesday, wednesday, thursday, friday" (wraps past Sunday).
    """

    if not isinstance(text, str):
        return text

    def _expand(m):

        a = DAY_MAP.get(m.group(1).lower())
        b = DAY_MAP.get(m.group(2).lower())

        if not a or not b:
            return m.group(0)

        i = ALL_DAYS.index(a)
        names = [_DAY_NAMES_FULL[a]]

        while ALL_DAYS[i] != b:
            i = (i + 1) % 7
            names.append(_DAY_NAMES_FULL[ALL_DAYS[i]])

        return ", ".join(names)

    return _DAY_RANGE_RE.sub(_expand, text)


def strip_date_phrases(text):
    """
    Remove durations and calendar dates ("for 10 days", "from 5 oct
    to 12 oct", "till friday") so the clock-time / day-of-week
    regexes never mistake them for times or weekdays.
    """

    if not isinstance(text, str):
        return text

    text = expand_day_ranges(text)

    for pat in _DATE_PATTERNS:
        text = pat.sub(" ", text)

    return text


def apply_duration_fallback(parsed, user_text):
    """
    The stated duration is applied by the server, never trusted to the
    model (it tends to turn "1 month" into 30 days, or "8 weeks" into
    56 sessions). If the user stated exactly one distinct duration, it
    is applied to every reminder without an explicit end date; "N days"
    and "N weeks" become N calendar days, "N months" calendar months.
    """

    phrases = set(find_duration_phrases(user_text))

    if len(phrases) != 1:
        return parsed

    count, unit = next(iter(phrases))

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
                for k in ("end_date", "end", "extend_days")
            ):
                continue

            item.pop("duration_days", None)
            item.pop("duration_months", None)

            if unit == "months":
                item["duration_months"] = count
            else:
                item["duration_days"] = count

    return parsed


_DATE_TOKEN = (
    r"(?:\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}(?:/\d{2,4})?"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+" + _MONTHS + r"(?:\s*,?\s*\d{4})?"
    r"|" + _MONTHS + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?"
    r"|day\s+after\s+tomorrow|tomorrow|today"
    r"|(?:this|next|coming)\s+" + _WEEKDAY_NAMES + r")"
)

_DATE_RANGE_RE = re.compile(
    r"(?:\b(?:from|between)\s+)?\b(?P<a>" + _DATE_TOKEN + r")\s*"
    r"(?:to|till|until|through|thru|and|-|\u2013|\u2014)\s*"
    r"(?P<b>" + _DATE_TOKEN + r")\b",
    re.IGNORECASE
)

_MONTH_NUM = {
    m: i for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"], 1
    )
}


def parse_one_date(tok, today, after=None):
    """
    One date phrase -> date. A day/month with no year is the next such
    day on or after `after` (default today). d/m/y order (India).
    """

    t = re.sub(r"\s+", " ", tok.strip().lower()).replace(",", "")
    floor = after or today

    if t == "today":
        return today

    if t == "tomorrow":
        return today + timedelta(days=1)

    if t.startswith("day after"):
        return today + timedelta(days=2)

    m = re.match(r"(this|next|coming)\s+(\w+)$", t)

    if m:
        code = DAY_MAP.get(m.group(2))

        if not code:
            return None

        ahead = (ALL_DAYS.index(code) - today.weekday()) % 7

        if m.group(1) == "next" and ahead == 0:
            ahead = 7

        return today + timedelta(days=ahead)

    iso = parse_iso_date(t)

    if iso:
        return iso

    year = None

    m = re.match(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?$", t)

    if m:
        day, month = int(m.group(1)), int(m.group(2))
        year = int(m.group(3)) if m.group(3) else None

    else:

        m = re.match(
            r"(\d{1,2})(?:st|nd|rd|th)? ([a-z]+)(?: (\d{4}))?$", t
        ) or None

        if m:
            day, mon = int(m.group(1)), m.group(2)
            year = int(m.group(3)) if m.group(3) else None
        else:
            m = re.match(
                r"([a-z]+)\.? (\d{1,2})(?:st|nd|rd|th)?(?: (\d{4}))?$", t
            )

            if not m:
                return None

            mon, day = m.group(1), int(m.group(2))
            year = int(m.group(3)) if m.group(3) else None

        month = _MONTH_NUM.get(mon[:3])

        if not month:
            return None

    if year is not None and year < 100:
        year += 2000

    try:

        if year is not None:
            return date(year, month, day)

        d = date(floor.year, month, day)

        if d < floor:
            d = date(floor.year + 1, month, day)

        return d

    except ValueError:
        return None


def apply_date_range_from_text(parsed, user_text):
    """
    "from 12 oct to 20 oct", "between 5/11 and 9/11",
    "from tomorrow to next friday": the start and end dates come from
    the user's words (the model often gets the year wrong). Weekday
    ranges without "this/next" ("monday to friday") are day-of-week
    selections, not dates.
    """

    if not isinstance(user_text, str):
        return parsed

    matches = list(_DATE_RANGE_RE.finditer(user_text))

    if len(matches) != 1:
        return parsed

    m = matches[0]

    today = today_ist()

    start = parse_one_date(m.group("a"), today)

    end = parse_one_date(m.group("b"), today, after=start)

    if not start or not end:
        return parsed

    if end < start:
        # "from 20 dec to 5 jan"
        try:
            end = end.replace(year=end.year + 1)
        except ValueError:
            return parsed

    for key in ("tasks", "medication", "custom"):

        for item in parsed.get(key) or []:

            if not isinstance(item, dict):
                continue

            if normalize_action(item.get("action")) != "set":
                continue

            if item.get("date"):
                continue

            for k in (
                "duration_days", "duration_months", "extend_days",
                "start", "end"
            ):
                item.pop(k, None)

            item["start_date"] = start.isoformat()
            item["end_date"] = end.isoformat()

    parsed["_has_date_range"] = True

    return parsed


_ONCE_RE = re.compile(
    r"\b(?:once|one[\s-]*time|only\s+once|just\s+once)\b",
    re.IGNORECASE
)


def apply_once_on_weekday(parsed, user_text):
    """
    "once on Monday only" is ONE occurrence on the next Monday (today
    counts if it is Monday), not a 7-day window.
    """

    if not isinstance(user_text, str) or not _ONCE_RE.search(user_text):
        return parsed

    days = extract_days_from_text(user_text)

    if not days or len(days) != 1:
        return parsed

    today = today_ist()

    when = today + timedelta(
        days=(ALL_DAYS.index(days[0]) - today.weekday()) % 7
    )

    for key in ("tasks", "medication", "custom"):

        for item in parsed.get(key) or []:

            if not isinstance(item, dict):
                continue

            if normalize_action(item.get("action")) != "set":
                continue

            if any(item.get(k) for k in ("start_date", "end_date")):
                continue

            for k in (
                "duration_days", "duration_months", "extend_days",
                "start", "end"
            ):
                item.pop(k, None)

            item["days"] = days

            if key == "custom":
                item["date"] = when.isoformat()
                item.pop("days", None)
            else:
                item["start_date"] = item["end_date"] = when.isoformat()

    parsed["_has_date_range"] = True

    return parsed


_EDIT_RE = re.compile(
    r"\b(?:change|move|shift|reschedule|switch|replace|edit|modify|"
    r"update|extend|also|add|another|more|instead|make\s+it)\b",
    re.IGNORECASE
)


def apply_ongoing_daily(parsed, user_text):
    """
    A fresh request with no duration and no dates ("drink water every 3
    hours between 8am and 8pm", "... daily") is an open-ended reminder.
    It must not inherit the dates or weekdays of an older schedule, so
    such items are flagged "_ongoing" (and given every day unless days
    were stated). Edits ("change", "also", "add", "extend"...) keep the
    existing dates.
    """

    if not isinstance(user_text, str):
        return parsed

    if strip_date_phrases(user_text) != expand_day_ranges(user_text):
        return parsed

    if _EDIT_RE.search(user_text) and not _DAILY_RE.search(user_text):
        return parsed

    for key in ("tasks", "medication", "custom"):

        for item in parsed.get(key) or []:

            if not isinstance(item, dict):
                continue

            if normalize_action(item.get("action")) != "set":
                continue

            if item.get("date"):
                continue

            if item.get("merge_times") or item.get("replace_times"):
                continue

            if any(
                item.get(k)
                for k in (
                    "start_date", "end_date", "start", "end",
                    "extend_days"
                )
            ):
                continue

            item.pop("duration_days", None)
            item.pop("duration_months", None)

            if not item.get("days"):
                item["days"] = ALL_DAYS[:]

            item["_ongoing"] = True

    return parsed


_DAILY_RE = re.compile(
    r"\b(?:daily|every\s*day|everyday|each\s+day|every\s+single\s+day)\b",
    re.IGNORECASE
)


def date_fields(item):
    return {
        "start_date": parse_iso_date(
            item.get("start_date") or item.get("start")
        ),
        "end_date": parse_iso_date(
            item.get("end_date") or item.get("end")
        ),
        "duration_days": positive_int(item.get("duration_days")),
        "duration_months": positive_int(
            item.get("duration_months"), maximum=120
        ),
        "extend_days": positive_int(item.get("extend_days")),
        "ongoing": bool(item.get("_ongoing"))
    }


def resolve_item_dates(item, existing=None):
    """
    Work out (start, end) for a reminder / habit.

    - duration_days      -> start today, end = N calendar days later
- duration_months    -> start today, end = N calendar months later
    - start_date/end_date-> used as given (the missing side is filled)
    - extend_days        -> existing end date + N days
    - nothing stated     -> keep the existing dates, otherwise
                            today .. today + DEFAULT_DURATION_DAYS

    Returns (start_date, end_date, used_default, error_message).
    """

    today = today_ist()

    ex_start = ex_end = None

    if existing and existing.get("enabled"):

        ex_start = parse_iso_date(
            existing.get("start_date") or existing.get("start")
        )

        ex_end = parse_iso_date(
            existing.get("end_date") or existing.get("end")
        )

        # An expired schedule is not worth keeping
        if ex_end and ex_end < today:
            ex_start = ex_end = None

    start = item.get("start_date")
    end = item.get("end_date")
    n = item.get("duration_days")
    mo = item.get("duration_months")
    ext = item.get("extend_days")

    used_default = False

    if (
        item.get("ongoing")
        and not (n or mo or ext or start or end)
    ):

        # "daily" with no duration: open-ended, never inherits the
        # dates of an older schedule
        start = today
        end = today + timedelta(days=ONGOING_DAYS)

    elif ext:

        base_end = ex_end if ex_end else today

        start = start or ex_start or today
        end = base_end + timedelta(days=ext)

    elif n or mo:

        start = start or today
        end = end or span_end(start, n, mo or 0)

    elif start and not end:

        if ex_end and ex_end >= start:
            end = ex_end
        else:
            # no end given: open-ended
            end = start + timedelta(days=ONGOING_DAYS)
            item["ongoing"] = True

    elif end and not start:

        start = ex_start if ex_start and ex_start <= end else today

    elif not start and not end:

        if ex_start and ex_end:
            start, end = ex_start, ex_end
        else:
            # no duration given: open-ended ("Daily"), not a 7-day stop
            start = today
            end = today + timedelta(days=ONGOING_DAYS)
            item["ongoing"] = True

    if end < start:
        return None, None, False, (
            f"end date {fmt_date(end)} is before the start date "
            f"{fmt_date(start)}"
        )

    if end < today:
        return None, None, False, (
            f"end date {fmt_date(end)} is already in the past"
        )

    wanted = item.get("days")

    if wanted and set(wanted) != set(ALL_DAYS):

        span_days = min((end - start).days, 400)

        if not any(
            ALL_DAYS[(start + timedelta(days=i)).weekday()] in wanted
            for i in range(span_days + 1)
        ):
            return None, None, False, (
                f"there is no {fmt_days(wanted)} between "
                f"{fmt_date(start)} and {fmt_date(end)}"
            )

    return start, end, used_default, None


# ---------------------------------------------------------
# Label matching for habits / medicines
# ---------------------------------------------------------

# Words that describe the KIND of item, not which one it is
# ("Reading habit" is the habit called "Reading").
_KIND_WORDS = {
    "habit", "habits", "reminder", "reminders", "task", "tasks",
    "event", "events", "alarm", "alarms", "the", "my"
}

# A label that means "all of them" rather than one named item
_ALL_MEDS = {
    "medication", "medications", "medicine", "medicines", "meds", "med",
    "all", "all medication", "all medications", "all medicine",
    "all medicines", "all meds", "pills", "tablets", "dose", "doses"
}

_ALL_HABITS = {
    "habit", "habits", "all", "all habits", "custom", "reminder",
    "reminders", "all reminders", "all custom", "custom habits",
    "custom reminders"
}


def _norm_label(s):
    s = re.sub(r"[^a-z0-9 ]", " ", str(s or "").lower())
    s = re.sub(r"\s+", " ", s).strip()

    words = s.split()
    core = [w for w in words if w not in _KIND_WORDS]

    return " ".join(core or words)


def is_generic_label(label, generic):
    """True for 'Medication' / 'all habits': a whole category."""

    raw = re.sub(r"[^a-z0-9 ]", " ", str(label or "").lower())
    raw = re.sub(r"\s+", " ", raw).strip()

    stripped = " ".join(
        w for w in raw.split() if w not in ("reminder", "reminders")
    )

    return raw in generic or stripped in generic


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

    # ", and 5pm", "and 5pm", ", 5pm", "& 5pm", "5pm 6pm"
    separator = re.compile(
        r"\s*(?:[,;&]\s*)?(?:\band\b\s*)?(?:then\s+)?",
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

            # "10:00 AM for 15 minutes": a bare number followed by a
            # unit is a length, not another clock time
            if not match.group(3) and re.match(
                r"\s*(?:min|hour|hr|sec|day|week|month)",
                text_lower[match.end():]
            ):
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

            # "at 10:00 PM to 3:00 AM" is a window, not two exact times
            rng = re.match(
                r"\s*(?:to|till|until|-|\u2013|\u2014)\s*"
                r"(\d{1,2})(?:\s*[:.]\s*(\d{1,2}))?\s*(am|pm)?",
                text_lower[pos:]
            )

            if rng and (match.group(3) or rng.group(3) or match.group(2)):
                pos += rng.end()
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
    """
    Extract explicit day-of-week selections.

    Understands exclusions too: "not on sat and sun", "except
    sunday", "skip saturdays" -> every day MINUS those days.

    Returns None when no days are mentioned, and [] when the user
    excluded every day ("except monday through sunday").
    """

    if not isinstance(text, str):
        return None

    text_lower = expand_day_ranges(text.lower())

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
        result = [
            d for d in ALL_DAYS
            if d in positive and d not in excluded
        ]

    elif excluded:
        result = [
            d for d in ALL_DAYS
            if d not in excluded
        ]

    else:
        return None

    # [] (not None) means the user excluded every day
    return result


def drop_unschedulable_items(parsed):
    """
    The user gave clock times but every one was invalid ("25:00"). The
    model may have guessed a time anyway, so nothing that depended on
    a time is scheduled -- the rejection is reported in "invalid".
    """

    def keep(key, it):

        if not isinstance(it, dict):
            return False

        if normalize_action(it.get("action")) != "set":
            return True

        if key == "tasks":
            return bool(
                it.get("interval_minutes") is not None
                or it.get("start_time")
                or it.get("end_time")
            )

        return False

    for key in ("tasks", "medication", "custom"):
        parsed[key] = [
            it for it in parsed.get(key) or [] if keep(key, it)
        ]


# ---------------------------------------------------------
# Per-habit times: every habit gets the times of ITS OWN clause
# ---------------------------------------------------------

_CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:[.;\n]|\b(?:and|also|then)\b(?=\s+(?:also\s+)?"
    r"(?:set\s+|add\s+|create\s+|remind\s+me\s+)?(?:a\s+|the\s+)?"
    r"(?:habit|reminder|remind)\b))\s*",
    re.IGNORECASE
)

_LABEL_STOP = {
    "the", "and", "for", "habit", "habits", "reminder", "reminders",
    "task", "event", "set", "add", "new", "daily", "session"
}


def _label_anchor(label, text_lower):
    """Position of the habit's most distinctive word in the text."""

    words = [
        w for w in _norm_label(label).split()
        if len(w) >= 3 and w not in _LABEL_STOP
    ]

    for w in sorted(words, key=len, reverse=True):

        m = re.search(r"\b" + re.escape(w[:5]), text_lower)

        if m:
            return m.start()

    return None


def apply_custom_times_from_text(parsed, time_text):
    """
    A message with several habits ("Floss at 10pm ... and Skincare at
    10:15pm ...") must not share one set of times. Each habit's times
    are re-read from the part of the message that talks about it, so
    minutes are never dropped and a length ("for 15 minutes") never
    becomes a second alarm.
    """

    if not isinstance(time_text, str):
        return parsed

    customs = [
        c for c in parsed.get("custom") or []
        if isinstance(c, dict)
        and normalize_action(c.get("action")) == "set"
    ]

    if not customs:
        return parsed

    text_lower = time_text.lower()

    clauses = []
    last = 0

    for m in _CLAUSE_SPLIT_RE.finditer(text_lower):
        clauses.append((last, m.start()))
        last = m.end()

    clauses.append((last, len(text_lower)))

    only_habits = not (
        parsed.get("tasks") or parsed.get("medication")
    )

    anchored = []

    for c in customs:
        anchored.append((c, _label_anchor(c.get("label"), text_lower)))

    for c, pos in anchored:

        if pos is None:

            if len(customs) == 1 and only_habits:
                lo, hi = 0, len(text_lower)
            else:
                continue

        else:

            lo, hi = next(
                ((a, b) for a, b in clauses if a <= pos <= b),
                (0, len(text_lower))
            )

            sharing = [
                p for oc, p in anchored
                if oc is not c and p is not None and lo <= p <= hi
            ]

            if sharing:

                # Several habits in one clause. Keep the model's
                # times when the text backs them up; only when one
                # does not appear in the text, read this habit's part
                # (from its name up to the next habit's name).
                in_clause, _ = extract_explicit_times_from_text(
                    text_lower[lo:hi]
                )

                have = {
                    (t["h"], t["m"])
                    for t in parse_time_list(c.get("times"))
                }

                if have and have <= set(in_clause):
                    continue

                later = [p for p in sharing if p > pos]

                lo = pos

                if later:
                    hi = min(later)

        times, _ = extract_explicit_times_from_text(text_lower[lo:hi])

        if times:
            c["times"] = [f"{h:02d}:{m:02d}" for h, m in times]

    return parsed


# ---------------------------------------------------------
# Wrong-category guards
# ---------------------------------------------------------

_PLANT_WATER_RE = re.compile(
    r"\bwater(?:ing)?\s+(?:the\s+|my\s+|our\s+|all\s+)?"
    r"(?:plants?|flowers?|garden|lawn|trees?|herbs?|seedlings?|"
    r"saplings?|crops?|pots?|grass|bonsai)\b",
    re.IGNORECASE
)

_DRINK_CUE_RE = re.compile(
    r"\b(?:drink\w*|hydrat\w*|sip\w*|glass(?:es)?|bottle|ml|"
    r"litres?|liters?)\b",
    re.IGNORECASE
)

_MOVE_KEYS = (
    "times", "days", "duration_days", "duration_months", "start_date",
    "end_date", "start", "end", "extend_days", "action", "merge_times"
)


def fix_plant_watering(parsed, user_text):
    """
    "Water the plants every Tuesday" is a habit, not the hydration
    (drink water) reminder. Only the drinking cue decides.
    """

    if not isinstance(user_text, str):
        return parsed

    m = _PLANT_WATER_RE.search(user_text)

    if not m or _DRINK_CUE_RE.search(user_text):
        return parsed

    label = clean_label(m.group(0))

    customs = parsed.setdefault("custom", [])

    already = any(
        isinstance(c, dict)
        and re.search(r"plant|flower|garden|lawn|tree|herb|grass",
                      str(c.get("label") or ""), re.IGNORECASE)
        for c in customs
    )

    kept = []

    for t in parsed.get("tasks") or []:

        if not (isinstance(t, dict) and t.get("type") == "hydration"):
            kept.append(t)
            continue

        if already:
            continue

        item = {k: t[k] for k in _MOVE_KEYS if k in t}
        item["label"] = label

        customs.append(item)
        already = True

    parsed["tasks"] = kept

    return parsed


_MED_CUE_RE = re.compile(
    r"\b(?:medic\w*|meds?|pills?|tablets?|capsules?|doses?|dosage|"
    r"syrup|vitamins?|supplements?|insulin|antibiotics?|drugs?|mg|"
    r"prescription|painkillers?|inhaler|snooze)\b",
    re.IGNORECASE
)

_BUILTIN_CUE_RE = re.compile(
    r"\b(?:hydrat\w*|water|drink\w*|eyes?|stretch\w*|walk\w*|"
    r"meditat\w*|pomodoro|bottle)\b",
    re.IGNORECASE
)


def drop_phantom_medication(parsed, user_text):
    """
    "Change hydration from 10am to 11am" must never create a made-up
    "Morning dose" medicine: a medication that is not named anywhere in
    the message, when the message is about a built-in reminder and has
    no medication wording at all, is dropped.
    """

    if not isinstance(user_text, str):
        return parsed

    if _MED_CUE_RE.search(user_text):
        return parsed

    if not _BUILTIN_CUE_RE.search(user_text):
        return parsed

    low = user_text.lower()

    kept = []

    for m in parsed.get("medication") or []:

        if not isinstance(m, dict):
            continue

        if normalize_action(m.get("action")) != "set":
            kept.append(m)
            continue

        words = [
            w for w in _norm_label(m.get("label")).split()
            if len(w) >= 4 and w not in ("dose", "morning", "evening",
                                         "night", "afternoon",
                                         "medication", "medicine")
        ]

        if any(w[:5] in low for w in words):
            kept.append(m)

    parsed["medication"] = kept

    return parsed


# ---------------------------------------------------------
# "change hydration from 10am to 11am" is an edit, not a window
# ---------------------------------------------------------

_TIME_PIECE = r"\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm)?"

_CHANGE_RE = re.compile(
    r"\b(?:change|move|shift|reschedule|switch|replace|edit|modify|"
    r"update)\b(?P<mid>[^.;]*?)\bfrom\s+(?P<t1>" + _TIME_PIECE
    + r")\s+(?:to|into|->|\u2192)\s+(?P<t2>" + _TIME_PIECE + r")",
    re.IGNORECASE
)

_TYPE_CUES = (
    ("hydration", re.compile(r"hydrat|water|drink", re.I)),
    ("eye", re.compile(r"\beyes?\b", re.I)),
    ("stretch", re.compile(r"stretch", re.I)),
    ("walk", re.compile(r"\bwalk", re.I)),
    ("bottle_clean", re.compile(r"bottle", re.I)),
)


def apply_time_change(parsed, user_text):
    """
    Turns "change hydration from 10am to 11am" into a replace-this-time
    edit of the existing reminder. Returns True if it handled the text.
    """

    if not isinstance(user_text, str):
        return False

    if re.search(r"\bevery\b|\bhourly\b", user_text, re.IGNORECASE):
        return False

    m = _CHANGE_RE.search(user_text)

    if not m:
        return False

    task_type = next(
        (t for t, rx in _TYPE_CUES if rx.search(m.group("mid"))),
        None
    )

    if not task_type:
        return False

    def one(piece, other_ampm):

        pm = re.match(
            r"(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?",
            piece.strip(),
            re.IGNORECASE
        )

        value, _ = _parse_time_token(
            pm.group(1), pm.group(2), pm.group(3) or other_ampm
        )

        return value

    ap1 = re.search(r"(am|pm)\s*$", m.group("t1"), re.I)
    ap2 = re.search(r"(am|pm)\s*$", m.group("t2"), re.I)

    old = one(m.group("t1"), ap2.group(1) if ap2 else None)
    new = one(m.group("t2"), ap1.group(1) if ap1 else None)

    if not old or not new:
        return False

    tasks = parsed.get("tasks") or []

    base = next(
        (
            t for t in tasks
            if isinstance(t, dict) and t.get("type") == task_type
        ),
        {}
    )

    edit = {
        k: v for k, v in base.items()
        if k not in (
            "times", "interval_minutes", "start_time", "end_time",
            "merge_times"
        )
    }

    edit.update({
        "type": task_type,
        "action": "set",
        "replace_times": [(old, new)]
    })

    parsed["tasks"] = [
        t for t in tasks
        if not (isinstance(t, dict) and t.get("type") == task_type)
    ] + [edit]

    return True


# ---------------------------------------------------------
# "every N minutes/hours": the number comes from the user's words
# ---------------------------------------------------------

_INTERVAL_RE = re.compile(
    r"\bevery\s+(?P<n>\d+(?:\.\d+)?|an?|one|two|three|four|five|six|"
    r"eight|ten|twelve|half(?:\s+an?)?)\s*(?P<u>minutes?|mins?|hours?|"
    r"hrs?|hr)\b",
    re.IGNORECASE
)


def find_interval_minutes(text):
    """Distinct 'every N minutes/hours' values in the text (minutes)."""

    out = set()

    if not isinstance(text, str):
        return out

    if re.search(r"\bevery\s+hour\b|\bhourly\b", text, re.IGNORECASE):
        out.add(60)

    for m in _INTERVAL_RE.finditer(text):

        raw = m.group("n").lower()

        if raw.startswith("half"):
            n = 0.5
        elif raw.replace(".", "", 1).isdigit():
            n = float(raw)
        else:
            n = _NUMBER_WORDS.get(raw)

        if not n:
            continue

        mins = n * (60 if m.group("u").lower().startswith("h") else 1)

        if mins >= 1:
            out.add(int(round(mins)))

    return out


_VOLUME_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(ml|millilit(?:er|re)s?|l|lit(?:er|re)s?)\b",
    re.IGNORECASE
)


def extract_hydration_goal(text):
    """
    "drink 500ml of water ..." -> "Goal: 500 ml each time".
    Display only: the v6 device JSON has no volume field, so it is
    returned beside the config ("extras"), never inside it.
    """

    if not isinstance(text, str):
        return None

    if not re.search(r"water|drink|hydrat", text, re.IGNORECASE):
        return None

    m = _VOLUME_RE.search(text)

    if not m:
        return None

    value = float(m.group(1))

    ml = value * (1 if m.group(2).lower().startswith("m") else 1000)

    if ml <= 0:
        return None

    shown = f"{ml:g} ml" if ml < 1000 else f"{ml / 1000:g} L"

    daily = re.search(
        r"\b(?:daily|per\s+day|a\s+day|each\s+day|total|goal)\b",
        text,
        re.IGNORECASE
    )

    return f"Daily goal: {shown}" if daily else f"Goal: {shown} each time"


def apply_interval_from_text(parsed, user_text):
    """
    If the user stated exactly one repeat interval, interval tasks use
    it (a "5 minute stretch every 3 hours" must stay every 180 min).
    """

    found = find_interval_minutes(user_text)

    if len(found) != 1:
        return parsed

    minutes = next(iter(found))

    for t in parsed.get("tasks") or []:

        if not isinstance(t, dict):
            continue

        if t.get("type") not in ABSOLUTE_TASK_TYPES - {"bottle_clean"}:
            continue

        if normalize_action(t.get("action")) != "set":
            continue

        if t.get("interval_minutes") is not None:
            t["interval_minutes"] = minutes

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

        if invalid_tokens:
            drop_unschedulable_items(parsed)

        return parsed

    tasks = parsed.get("tasks") or []

    for task in tasks:

        if task.get("type") not in ABSOLUTE_TASK_TYPES:
            continue

        # Turning something off never needs a time
        if normalize_action(task.get("action")) != "set":
            continue

        task["times"] = [
            f"{h:02d}:{m:02d}"
            for h, m in explicit_times
        ]

        # "every 2 hours and also at 10:15": keep the interval, the
        # exact times are merged with it (and de-duplicated) later.
        if task.get("interval_minutes") is None:
            task.pop("start_time", None)
            task.pop("end_time", None)

    if not parsed.get("active_window"):
        parsed["active_window"] = {
            "start": "00:00",
            "end": "23:59"
        }

    return parsed


def apply_medication_extras(plan, user_text):
    """
    If the user stated a snooze period ("15 mins snooze"), attach it
    to the medication entries. In the v6 schema snooze_min is a single
    global value under reminders.medication.
    """

    stated_snooze = extract_stated_snooze_minutes(user_text)

    if stated_snooze is None:
        return plan

    for med in plan.get("medication", []) or []:
        med["snooze_min"] = stated_snooze

    return plan


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

    tail = (
        r"(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?\s*"
    )

    patterns = [
        # "from 9am to 6pm" / "between 8am and 8pm"
        re.compile(
            r"(?:from|between)\s+" + tail
            + r"(?:to|and|till|until|-|\u2013|\u2014)\s*"
            + r"(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?",
            re.IGNORECASE
        ),
        # "at 10pm to 3am" (an "at" that runs to an end time)
        re.compile(
            r"\bat\s+" + tail
            + r"(?:to|till|until|-|\u2013|\u2014)\s*"
            + r"(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?",
            re.IGNORECASE
        )
    ]

    match = next(
        (m for m in (pt.search(user_text) for pt in patterns) if m),
        None
    )

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

            # an interval task may also carry extra exact times
            # ("every 2 hours ... and also at 10:15")
            if task.get("interval_minutes") is None:
                task.pop("times", None)
            else:
                # "at 10pm to 3am every hour": the model may also list
                # the window ends as exact times - they are not extras
                ends = {
                    f"{sh:02d}:{sm:02d}", f"{eh:02d}:{em:02d}"
                }

                extra = [
                    x for x in task.get("times") or []
                    if x not in ends
                ]

                if extra:
                    task["times"] = extra
                else:
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

    # "5 minute stretch every 3 hours from 9am to 6pm": the window is
    # the span the repeats happen in, the 5 minutes is the length of
    # one break. They are not meant to match.
    if find_interval_minutes(user_text):
        return []

    mismatches = []

    for t in plan.get("tasks", []):

        if t.get("interval_minutes") is not None:
            continue

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

    If every day was excluded there is nothing left to schedule: the
    items are dropped and a reason is stashed on parsed["_invalid_days"].
    """

    extracted = extract_days_from_text(user_text)

    if extracted is None:
        return parsed

    if not extracted:

        parsed["_invalid_days"] = [{
            "task": "days",
            "reason": (
                "every day of the week was excluded, so there is "
                "nothing left to schedule"
            )
        }]

        for key in ("tasks", "medication", "custom"):
            parsed[key] = [
                it for it in parsed.get(key) or []
                if not isinstance(it, dict)
                or normalize_action(it.get("action")) != "set"
            ]

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

        entry = {
            "type": task_type,
            "action": normalize_action(t.get("action")),
            "mode": mode,
            "interval_minutes": interval_minutes,
            "interval_days": positive_int(
                t.get("interval_days"),
                maximum=365
            ),
            "times": parsed_times,
            "merge_times": bool(t.get("merge_times")),
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
            "days": norm_days_opt(
                t.get("days")
            ),
            "replace_times": [
                (tuple(o), tuple(n))
                for o, n in (t.get("replace_times") or [])
            ]
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
            "label": clean_label(
                m.get("label"),
                "Medication"
            ),
            "new_label": clean_label(
                m.get("new_label")
            ),
            "action": normalize_action(
                m.get("action")
            ),
            "doses": parse_time_list(
                m.get("times")
            ),
            "merge_times": bool(m.get("merge_times")),
            "days": norm_days_opt(
                m.get("days")
            )
        }

        entry.update(date_fields(m))

        out.append(entry)

    return out


def normalize_custom(parsed):
    """
    Custom habits / reminders -- anything that isn't hydration, eye,
    stretch, walk, meditation, pomodoro, bottle cleaning or
    medication (e.g. "play chess at 6pm for 10 days").

    A one-time reminder carries "date"; it is stored with
    start_date == end_date so the device only fires on that day.
    """

    items = parsed.get("custom") or []

    out = []

    for c in items:

        if not isinstance(c, dict):
            continue

        times = parse_time_list(c.get("times"))

        entry = {
            "label": clean_label(
                c.get("label"),
                "Reminder"
            ),
            "new_label": clean_label(
                c.get("new_label")
            ),
            "action": normalize_action(
                c.get("action")
            ),
            "times": times,
            "merge_times": bool(c.get("merge_times")),
            "days": norm_days_opt(
                c.get("days")
            ),
            "one_time_date": parse_iso_date(
                c.get("date")
            )
        }

        entry.update(date_fields(c))

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
# V6 CONFIG (new JSON) - BASE TEMPLATE
# =========================================================
#
# Layout follows frost-config.json (schema_ver 6):
#
#   _meta
#   reminders
#     hydration / stretch / eye / walk   (start_date, end_date, days,
#                                          abs.times [{h,m}], ...)
#     bottle_clean                       (start_date, end_date,
#                                          interval_days, time {h,m})
#     meditation                         (start_date, end_date, days,
#                                          times [{start,end}])
#     medication.medicines[]             (start / end, days, doses)
#     custom.events[]                    (HABITS: start_date, end_date,
#                                          days, times [{h,m}])
#   audio
#   pomodoro                             (start_date, end_date, days,
#                                          laps [{start, cycles}])

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
    """
    Fresh v6 config. Every reminder starts disabled; its dates are
    pre-filled with today .. today + DEFAULT_DURATION_DAYS so the
    schema is always complete and valid.
    """

    today = today_ist()

    s = today.isoformat()
    e = span_end(today, DEFAULT_DURATION_DAYS).isoformat()

    def basic():

        return {
            "enabled": False,
            "start_date": s,
            "end_date": e,
            "days": ALL_DAYS[:],
            "abs": {
                "times": []
            },
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
    """
    Start from the config the frontend currently holds (so the
    chatbot can ADD to / CHANGE / REMOVE things) or, if none was
    sent, from a fresh base. Missing sections are filled in.
    """

    base = make_base_config()

    if not (
        isinstance(current, dict)
        and isinstance(current.get("reminders"), dict)
    ):
        return base

    cfg = copy.deepcopy(current)

    # Old v6 files kept bottle_clean at the top level (hour/minute).
    legacy = cfg.pop("bottle_clean", None)

    if (
        isinstance(legacy, dict)
        and "bottle_clean" not in cfg["reminders"]
    ):
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
    """Working state while one plan is applied to a config."""

    def __init__(self, config, global_start, global_end):
        self.config = config
        self.reminders = config["reminders"]
        self.gs = global_start
        self.ge = global_end
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
    """True if an enabled basic reminder already has exact times."""

    return bool(
        cur.get("enabled")
        and (cur.get("abs") or {}).get("times")
    )


def expand_interval_times(sh, sm, eh, em, interval_min):
    """
    'every 2 hours from 9am to 5pm' -> exact times 9,11,13,15,17.
    A window may run past midnight: 11pm -> 3am every 2 hours is
    23:00, 01:00, 03:00.
    """

    span = window_duration_minutes(sh, sm, eh, em) or 0

    first = sh * 60 + sm

    out = []

    offset = 0

    while offset <= span:

        cur = (first + offset) % (24 * 60)

        out.append({"h": cur // 60, "m": cur % 60})

        offset += max(1, interval_min)

    return out


def dedupe_times(times):
    """Same clock time listed twice -> once, first position kept."""

    out = []
    seen = set()

    for t in times or []:

        key = (t.get("h"), t.get("m"))

        if key in seen:
            continue

        seen.add(key)
        out.append({"h": t["h"], "m": t["m"]})

    return out


def _ask_basic(ctx, key):
    name = DISPLAY_NAMES[key].lower()

    ctx.need_time(
        key,
        f"What time should I set for your {name}? Give exact times "
        f"(e.g. \"at 10:15am and 2pm\") or an interval "
        f"(e.g. \"every 30 minutes from 9am to 5pm\")."
    )


# ---------------------------------------------------------
# Windows that cross midnight
# ---------------------------------------------------------
#
# One reminder has ONE list of times and ONE date range, so 12-3 AM
# would fire on the same dates as 9-11 PM. For "9pm to 3am" the after-
# midnight times belong to the NEXT date, so they are saved as a
# separate habit (custom.events - an existing structure) whose dates
# and days are one day later.

SHORT_NAMES = {
    "hydration": "Drink water",
    "eye": "Eye break",
    "stretch": "Stretch",
    "walk": "Walk"
}

AFTER_MIDNIGHT = " (after midnight)"


def after_midnight_label(key):
    return SHORT_NAMES[key] + AFTER_MIDNIGHT


def shift_days_next(days):
    """Mon, Fri -> Tue, Sat (the day after each chosen day)."""

    return normalize_days([
        ALL_DAYS[(ALL_DAYS.index(d) + 1) % 7]
        for d in normalize_days(days)
    ])


def _sync_custom_enabled(ctx):

    cust = ctx.reminders["custom"]

    cust["enabled"] = any(
        e.get("enabled") for e in cust.get("events", [])
    )


def remove_after_midnight(ctx, key):

    cust = ctx.reminders["custom"]

    label = after_midnight_label(key)

    events = cust.setdefault("events", [])

    kept = [e for e in events if e.get("label") != label]

    if len(kept) != len(events):
        cust["events"] = kept
        _sync_custom_enabled(ctx)


def put_after_midnight(ctx, key, times, start_d, end_d, days):

    cust = ctx.reminders["custom"]

    events = cust.setdefault("events", [])

    label = after_midnight_label(key)

    next_start = start_d + timedelta(days=1)
    next_end = end_d + timedelta(days=1)
    next_days = shift_days_next(days)

    match = next(
        (e for e in events if e.get("label") == label), None
    )

    if match:

        match.update({
            "enabled": True,
            "start_date": next_start.isoformat(),
            "end_date": next_end.isoformat(),
            "days": next_days,
            "times": times
        })

    else:

        events.append({
            "id": next_item_id(events, "custom"),
            "label": label,
            "enabled": True,
            "start_date": next_start.isoformat(),
            "end_date": next_end.isoformat(),
            "display_ms": cust.get("display_ms", 60000),
            "days": next_days,
            "times": times,
            **CUSTOM_TEXT_DEFAULTS
        })

    _sync_custom_enabled(ctx)

    return next_start, next_end, next_days


def _apply_simple_task(ctx, t, key, default_interval_min):
    """hydration / eye / stretch / walk -- always exact times (abs)."""

    cur = ctx.reminders[key]
    name = DISPLAY_NAMES[key]

    if t["action"] in ("disable", "remove"):
        cur["enabled"] = False
        remove_after_midnight(ctx, key)
        ctx.summary.append(f"{name}: turned off")
        return

    start_d, end_d, used_default, err = resolve_item_dates(t, cur)

    if err:
        ctx.invalid.append({
            "task": key,
            "reason": f"{name}: {err}"
        })
        return

    days = t.get("days") or (
        normalize_days(cur.get("days"))
        if cur.get("enabled")
        else ALL_DAYS[:]
    )

    window_first = None   # minute-of-day the window starts (if used)
    keeps_times = False

    sh, sm = t.get("start_time") or ctx.gs or (0, 0)
    eh, em = t.get("end_time") or ctx.ge or (23, 59)

    wants_interval = (
        t.get("interval_minutes") is not None
        or t.get("start_time") is not None
        or t.get("end_time") is not None
    )

    def expand_window():
        """Interval -> exact times, or None after recording why not."""

        span = window_duration_minutes(sh, sm, eh, em)

        interval_min = t.get("interval_minutes")

        if not interval_min or interval_min <= 0:
            interval_min = default_interval_min

        if span is None or span <= 0:

            ctx.invalid.append({
                "task": key,
                "reason":
                    f"{key} window {sh:02d}:{sm:02d}→"
                    f"{eh:02d}:{em:02d} is zero-length "
                    f"(start and end are the same time)"
            })

            return None, interval_min

        if interval_min > span:

            ctx.invalid.append({
                "task": key,
                "reason":
                    f"{key}: every {interval_min} min doesn't fit "
                    f"in the {span}-min window "
                    f"{sh:02d}:{sm:02d}→{eh:02d}:{em:02d}"
            })

            return None, interval_min

        return (
            expand_interval_times(sh, sm, eh, em, interval_min),
            interval_min
        )

    def too_many(times, interval_min):

        if len(times) <= MAX_ABSOLUTE_TIMES:
            return False

        ctx.invalid.append({
            "task": key,
            "reason":
                f"{key}: this makes {len(times)} reminders, but at "
                f"most {MAX_ABSOLUTE_TIMES} exact times are allowed - "
                f"use a longer gap or a shorter window"
        })

        return True

    if t.get("replace_times"):

        # "change hydration from 10am to 11am"
        existing = (
            list((cur.get("abs") or {}).get("times") or [])
            if cur.get("enabled")
            else []
        )

        times = existing

        for (oh, om), (nh, nm) in t["replace_times"]:

            old_t = {"h": oh, "m": om}
            new_t = {"h": nh, "m": nm}

            if old_t in times:
                times = [new_t if x == old_t else x for x in times]

            elif existing:

                ctx.invalid.append({
                    "task": key,
                    "reason":
                        f"{name}: there is no reminder at "
                        f"{fmt_time12(oh, om)} to change"
                })

                return

            else:
                times = times + [new_t]

        times = dedupe_times(times)

        what = fmt_times(times)

    elif t.get("times") and t.get("interval_minutes") is not None:

        # "every 2 hours and also at 10:15": interval + exact times,
        # merged and de-duplicated
        window_times, interval_min = expand_window()

        if window_times is None:
            return

        first = sh * 60 + sm

        window_first = first

        times = sorted(
            dedupe_times(window_times + list(t["times"])),
            key=lambda x: (x["h"] * 60 + x["m"] - first) % (24 * 60)
        )

        if too_many(times, interval_min):
            return

        what = f"{fmt_times(times)} (every {interval_min} min + exact)"

    elif t.get("times"):

        times = t["times"]

        if t.get("merge_times"):
            times = merge_time_lists(
                (cur.get("abs") or {}).get("times"),
                times
            )

        times = dedupe_times(times)

        what = fmt_times(times)

    elif wants_interval:

        # The v6 JSON only stores exact times, so "every X minutes
        # from A to B" is expanded into the list of exact times.

        times, interval_min = expand_window()

        if times is None:
            return

        if too_many(times, interval_min):
            return

        window_first = sh * 60 + sm

        what = f"{fmt_times(times)} (every {interval_min} min)"

    elif has_schedule(cur):

        # Only dates / days are changing - keep the exact times
        times = (cur.get("abs") or {}).get("times")

        keeps_times = True

        what = fmt_times(times)

    else:

        _ask_basic(ctx, key)
        return

    # Split a window that crosses midnight: evening times stay here,
    # the after-midnight times go to the next date.
    after_times = []

    if window_first is not None:

        evening, late = [], []

        for x in times:

            offset = (x["h"] * 60 + x["m"] - window_first) % (24 * 60)

            (late if window_first + offset >= 24 * 60 else evening).append(x)

        if late and evening:

            all_text = fmt_times(times)

            suffix = what[len(all_text):] if what.startswith(all_text) else ""

            times = evening
            after_times = late
            what = fmt_times(times) + suffix

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

    ctx.summary.append(
        f"{name}: {what} · "
        f"{fmt_when(days, start_d, end_d, t.get('ongoing'))}"
    )

    if after_times:

        n_start, n_end, n_days = put_after_midnight(
            ctx, key, after_times, start_d, end_d, days
        )

        ctx.summary.append(
            f"{name} after midnight (next date): "
            f"{fmt_times(after_times)} · "
            f"{fmt_when(n_days, n_start, n_end, t.get('ongoing'))}"
        )

        ctx.notes.append(
            f"{name}: the times after midnight "
            f"({fmt_times(after_times)}) belong to the next date, so "
            f"they are saved as the habit "
            f"'{after_midnight_label(key)}' one day later."
        )

    elif keeps_times:

        # only dates/days changed: keep an existing after-midnight
        # companion in step with them
        label = after_midnight_label(key)

        companion = next(
            (
                e for e in ctx.reminders["custom"].get("events", [])
                if e.get("label") == label
            ),
            None
        )

        if companion:
            put_after_midnight(
                ctx, key, companion.get("times", []),
                start_d, end_d, days
            )

    else:

        remove_after_midnight(ctx, key)

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
        ctx.invalid.append({
            "task": "meditation",
            "reason": f"{name}: {err}"
        })
        return

    windows = []
    spans = []
    bad_window = False

    for t in active:

        if not (t.get("start_time") and t.get("end_time")):
            continue

        sh, sm = t["start_time"]
        eh, em = t["end_time"]

        span = window_duration_minutes(sh, sm, eh, em)

        if span is None or span <= 0:

            bad_window = True

            ctx.invalid.append({
                "task": "meditation",
                "reason":
                    f"meditation window {sh:02d}:{sm:02d}→"
                    f"{eh:02d}:{em:02d} is zero-length "
                    f"(start and end are the same time)"
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

        display_ms = max(spans) * 60000

    elif bad_window:
        return

    elif existing_windows:

        windows = existing_windows
        display_ms = cur.get("display_ms", 600000)

    else:

        ctx.need_time(
            "meditation",
            "What start and end time should I set for meditation? "
            "(e.g. \"from 7am to 7:30am\")"
        )

        return

    days = first.get("days") or (
        normalize_days(cur.get("days"))
        if cur.get("enabled")
        else ALL_DAYS[:]
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
        f"{fmt_time12(w['start']['h'], w['start']['m'])}–"
        f"{fmt_time12(w['end']['h'], w['end']['m'])}"
        for w in windows
    )

    ctx.summary.append(
        f"{name}: {what} · "
        f"{fmt_when(days, start_d, end_d, first.get('ongoing'))}"
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
        ctx.invalid.append({
            "task": "pomodoro",
            "reason": f"{name}: {err}"
        })
        return

    was_on = bool(pomo.get("enabled"))

    focus = t.get("focus_min") or (
        pomo.get("focus_min") if was_on else None
    ) or 25

    brk = t.get("break_min") or (
        pomo.get("break_min") if was_on else None
    ) or 5

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

            span = window_duration_minutes(sh, sm, eh, em)

            if span is None or span <= 0:

                ctx.invalid.append({
                    "task": "pomodoro",
                    "reason":
                        f"pomodoro window {sh:02d}:{sm:02d}→"
                        f"{eh:02d}:{em:02d} is zero-length "
                        f"(start and end are the same time)"
                })

                return

            fit = (span + brk) // (focus + brk)

            if cycles is None:

                if fit < 1:

                    ctx.invalid.append({
                        "task": "pomodoro",
                        "reason":
                            f"pomodoro: a {focus}-min focus session "
                            f"doesn't fit in the {span}-min window "
                            f"{sh:02d}:{sm:02d}→{eh:02d}:{em:02d}"
                    })

                    return

                cycles = fit

            else:

                needed = focus * cycles + brk * max(cycles - 1, 0)

                if needed > span:

                    ctx.invalid.append({
                        "task": "pomodoro",
                        "reason":
                            f"pomodoro: {cycles} cycles of "
                            f"{focus}/{brk} min need {needed} min "
                            f"but the window "
                            f"{sh:02d}:{sm:02d}→{eh:02d}:{em:02d} "
                            f"is only {span} min"
                    })

                    return

        if cycles is None:
            cycles = (
                existing_laps[0].get("cycles")
                if existing_laps
                else 4
            ) or 4

        laps = [
            {
                "start": {"h": h, "m": m},
                "cycles": cycles
            }
            for h, m in starts
        ]

    elif existing_laps:

        laps = existing_laps

    else:

        ctx.need_time(
            "pomodoro",
            "What time should the pomodoro start? "
            "(e.g. \"at 9am\")"
        )

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
        "focus_counter": pomo.get(
            "focus_counter", dict(COUNTER_DEFAULTS)
        ),
        "break_counter": pomo.get(
            "break_counter", dict(COUNTER_DEFAULTS)
        )
    }

    for k, v in pomo.items():
        if k not in new and k not in (
            "cycles", "lap_mode_enabled",
            "auto_start_break", "auto_start_focus"
        ):
            new[k] = v

    ctx.config["pomodoro"] = new

    what = ", ".join(
        f"{fmt_time12(l['start']['h'], l['start']['m'])} "
        f"({l['cycles']} cycles)"
        for l in laps
    )

    ctx.summary.append(
        f"{name} {focus}/{brk}: {what} · "
        f"{fmt_when(days, start_d, end_d, t.get('ongoing'))}"
    )

    if used_default:
        ctx.default_note(name, end_d)


def _apply_bottle_clean(ctx, t):

    cur = ctx.reminders["bottle_clean"]
    name = DISPLAY_NAMES["bottle_clean"]

    if t["action"] in ("disable", "remove"):
        cur["enabled"] = False
        ctx.summary.append(f"{name}: turned off")
        return

    if t.get("days") and set(t["days"]) != set(ALL_DAYS):

        # bottle_clean only stores "every N days" - the JSON has no
        # weekday field and is not changed. Say so instead of silently
        # saving a daily reminder.
        ctx.invalid.append({
            "task": "bottle_clean",
            "reason":
                f"{name} can only repeat every N days, not on specific "
                f"weekdays ({fmt_days(t['days'])}) - try \"every 7 "
                f"days\" or \"every day\""
        })

        return

    start_d, end_d, used_default, err = resolve_item_dates(t, cur)

    if err:
        ctx.invalid.append({
            "task": "bottle_clean",
            "reason": f"{name}: {err}"
        })
        return

    was_on = bool(cur.get("enabled"))

    if t.get("times"):
        time_obj = dict(t["times"][0])

    elif was_on and cur.get("time"):
        time_obj = cur["time"]

    else:

        ctx.need_time(
            "bottle_clean",
            "What time of day should I remind you to clean your "
            "bottle? (e.g. \"at 6pm\")"
        )

        return

    interval_days = (
        t.get("interval_days")
        or (cur.get("interval_days") if was_on else None)
        or 1
    )

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

    how_often = (
        "every day"
        if interval_days == 1
        else f"every {interval_days} days"
    )

    if t.get("ongoing") and interval_days == 1:
        when = "Daily"
    else:
        when = (
            f"{how_often} · "
            f"{fmt_date(start_d)} → {fmt_date(end_d)}"
        )

    ctx.summary.append(
        f"{name}: {fmt_time12(time_obj['h'], time_obj['m'])} · {when}"
    )

    if used_default:
        ctx.default_note(name, end_d)


def _apply_medication(ctx, plan_meds):

    med_cfg = ctx.reminders["medication"]
    medicines = med_cfg.setdefault("medicines", [])

    for m in plan_meds:

        if m.get("_invalid"):
            continue

        label = m["label"]
        match = find_by_label(medicines, label)

        if (
            m["action"] in ("disable", "remove")
            and not match
            and is_generic_label(label, _ALL_MEDS)
        ):

            # "cancel all medication reminders" -> the whole category
            if m["action"] == "remove":
                medicines.clear()
                ctx.summary.append("Medication reminders: all removed")
            else:
                for med in medicines:
                    med["enabled"] = False
                ctx.summary.append("Medication reminders: all turned off")

            continue

        if m["action"] in ("disable", "remove"):

            if not match:
                ctx.invalid.append({
                    "task": "medication",
                    "reason": f"no medication named '{label}' found"
                })
                continue

            if m["action"] == "remove":
                medicines.remove(match)
                ctx.summary.append(f"Medication '{label}': removed")
            else:
                match["enabled"] = False
                ctx.summary.append(f"Medication '{label}': turned off")

            continue

        start_d, end_d, used_default, err = resolve_item_dates(m, match)

        if err:
            ctx.invalid.append({
                "task": "medication",
                "reason": f"medication '{label}': {err}"
            })
            continue

        doses = m["doses"]

        if doses and match and m.get("merge_times"):
            doses = merge_time_lists(match.get("doses"), doses)

        if not doses:

            if match and match.get("doses"):
                doses = match["doses"]

            else:

                ctx.need_time(
                    "medication",
                    f"What time(s) should I remind you to take "
                    f"{label}? (e.g. \"at 8am and 8pm\")"
                )

                continue

        days = m.get("days") or (
            normalize_days(match.get("days")) if match else ALL_DAYS[:]
        )

        if match:

            match.update({
                "label": m.get("new_label") or match.get("label"),
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
            f"Medication '{shown}': {fmt_times(doses)} · "
            f"{fmt_when(days, start_d, end_d, m.get('ongoing'))}"
        )

        if used_default:
            ctx.default_note(f"medication '{shown}'", end_d)

        if m.get("snooze_min") is not None:
            med_cfg["snooze_min"] = m["snooze_min"]

    med_cfg["enabled"] = any(
        x.get("enabled") for x in medicines
    )


def _apply_custom(ctx, plan_custom):
    """Habits: 'I want to play chess at 6pm for 10 days'."""

    cust = ctx.reminders["custom"]
    events = cust.setdefault("events", [])

    for c in plan_custom:

        label = c["label"]
        match = find_by_label(events, label)

        if (
            c["action"] in ("disable", "remove")
            and not match
            and is_generic_label(label, _ALL_HABITS)
        ):

            if c["action"] == "remove":
                events.clear()
                ctx.summary.append("Habits: all removed")
            else:
                for ev in events:
                    ev["enabled"] = False
                ctx.summary.append("Habits: all turned off")

            continue

        if c["action"] in ("disable", "remove"):

            if not match:
                ctx.invalid.append({
                    "task": "custom",
                    "reason": f"no habit named '{label}' found"
                })
                continue

            if c["action"] == "remove":
                events.remove(match)
                ctx.summary.append(f"Habit '{label}': removed")
            else:
                match["enabled"] = False
                ctx.summary.append(f"Habit '{label}': turned off")

            continue

        times = c["times"]

        if times and match and c.get("merge_times"):
            times = merge_time_lists(match.get("times"), times)

        if not times:

            if match and match.get("times"):
                times = match["times"]

            else:

                ctx.need_time(
                    "custom",
                    f"What time should I remind you for "
                    f"\"{label}\"? (e.g. \"at 6pm\" - you can give "
                    f"more than one time)"
                )

                continue

        used_default = False

        if c.get("one_time_date"):

            one = c["one_time_date"]

            if one < today_ist():
                ctx.invalid.append({
                    "task": "custom",
                    "reason":
                        f"habit '{label}': {fmt_date(one)} is "
                        f"already in the past"
                })
                continue

            start_d = end_d = one
            days = ALL_DAYS[:]

        else:

            start_d, end_d, used_default, err = resolve_item_dates(
                c, match
            )

            if err:
                ctx.invalid.append({
                    "task": "custom",
                    "reason": f"habit '{label}': {err}"
                })
                continue

            days = c.get("days") or (
                normalize_days(match.get("days"))
                if match
                else ALL_DAYS[:]
            )

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

        when = fmt_when(days, start_d, end_d, c.get("ongoing"))

        ctx.summary.append(
            f"Habit '{shown}': {fmt_times(times)} · {when}"
        )

        if used_default:
            ctx.default_note(f"'{shown}'", end_d)

    cust["enabled"] = any(
        e.get("enabled") for e in events
    )


def convert_to_new_schema(plan, current_config=None):
    """
    Apply a normalized plan to the v6 config.

    Returns (config, invalid, missing, summary, notes)
      invalid - things the user asked for that cannot work
      missing - things we must ask the user about (usually a time)
    """

    config = load_working_config(current_config)

    active = plan.get(
        "active_window"
    ) or {
        "start": "00:00",
        "end": "23:59"
    }

    ctx = Ctx(
        config,
        parse_time(active.get("start") or "00:00"),
        parse_time(active.get("end") or "23:59")
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
# FOLLOW-UP QUESTIONS (missing time etc.)
# =========================================================
#
# /parse is stateless. When something is missing (usually the time),
# the response carries   "pending": {"text": "..."}   and the
# frontend must send that object back with the user's next message:
#
#     { "text": "at 10:15am and 2pm", "pending": <object from reply> }
#
# The server joins the original request and the answer, so
# "i want eye break for 10 days"  +  "at 10:15am and 2pm"
# is parsed as one complete instruction.

CANCEL_RE = re.compile(
    r"^\s*(?:cancel|never\s*mind|nevermind|forget\s+it|"
    r"skip(?:\s+it)?|no\s+thanks?)\s*[.!]*\s*$",
    re.IGNORECASE
)


def merge_pending(text, pending):
    """
    Returns (combined_text, cancelled).
    """

    text = (text or "").strip()

    if not (isinstance(pending, dict) and pending.get("text")):
        return text, False

    if CANCEL_RE.match(text):
        return text, True

    answer = text

    # "10:15am and 2pm"  ->  "at 10:15am and 2pm"
    if re.match(r"^\d", answer) and not re.search(
        r"\bat\b", answer, re.IGNORECASE
    ):
        answer = "at " + answer

    combined = f"{str(pending['text']).strip()}. {answer}"

    return combined[:2000], False


def build_question_reply(missing):

    if len(missing) == 1:
        return missing[0]["question"]

    lines = [
        f"{i}. {m['question']}"
        for i, m in enumerate(missing, 1)
    ]

    return (
        "I need a little more information before I can save this:\n"
        + "\n".join(lines)
    )


# =========================================================
# ROUTE
# =========================================================

@app.route("/parse", methods=["POST"])
def parse_schedule():

    logs = []

    start_time = time.perf_counter()

    try:

        data = request.get_json() or {}

        logs.append(
            "Step 1: Input received"
        )

        # Join with the earlier request if we asked a question
        user_text, cancelled = merge_pending(
            data.get("text", ""),
            data.get("pending")
        )

        if cancelled:

            logs.append(
                "Step 1b: Pending request cancelled by user"
            )

            return jsonify({
                "reply": "No problem - I've dropped that request.",
                "logs": logs
            })

        if data.get("pending"):
            logs.append(
                "Step 1b: Follow-up answer merged with earlier request"
            )

        today_ist_dt = datetime.now(IST)

        dated_system_prompt = (
            SYSTEM_PROMPT
            + "\n\nTODAY'S DATE: "
            + today_ist_dt.strftime("%Y-%m-%d (%A)")
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
                    "content": user_text
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

        # Text with durations / calendar dates removed, so
        # "for 10 days" or "from 5 oct to 12 oct" can never be
        # mistaken for a clock time, a window or a weekday.
        time_text = strip_date_phrases(user_text)

        # Category guards: "water the plants" is not hydration, and
        # a hydration edit must not invent a medicine
        parsed = fix_plant_watering(parsed, user_text)

        parsed = drop_phantom_medication(parsed, user_text)

        # "change hydration from 10am to 11am" is an edit of an
        # existing time, not a 10am -> 11am window
        is_time_change = apply_time_change(parsed, time_text)

        # "every 3 hours" is taken from the user's own words
        parsed = apply_interval_from_text(parsed, time_text)

        time_parse_invalid = []

        if not is_time_change:

            # Force exact "at <time>"

            parsed = force_absolute_times_from_user_text(
                parsed,
                time_text
            )

            time_parse_invalid = parsed.pop("_invalid_times", [])

            # Force "from X to Y" / "between X and Y" into a window
            parsed = force_window_from_text(
                parsed,
                time_text
            )

        # Every habit keeps the times of its own part of the message
        parsed = apply_custom_times_from_text(parsed, time_text)

        # Apply day rules

        parsed = apply_days_from_user_text(
            parsed,
            time_text
        )

        invalid_days = parsed.pop("_invalid_days", [])

        # "once on Monday" and "from 12 oct to 20 oct"
        parsed = apply_once_on_weekday(parsed, time_text)

        parsed = apply_date_range_from_text(parsed, user_text)

        has_dates = parsed.pop("_has_date_range", False)

        # Duration ("for 10 days", "for 2 months") is calendar time

        parsed = apply_duration_fallback(
            parsed,
            user_text
        )

        # "... daily" with no duration is open-ended

        parsed = apply_ongoing_daily(
            parsed,
            user_text
        )

        logs.append(
            "Step 3: JSON parsed, exact-time, day and duration rules applied"
        )

        plan = build_plan(parsed)

        plan = apply_medication_extras(
            plan,
            time_text
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
            f"Step 4: Tasks={len(plan['tasks'])}, "
            f"Meds={len(plan['medication'])}, "
            f"Habits={len(plan['custom'])}"
        )

        # Stated-duration vs actual-window validation

        duration_mismatches = check_duration_mismatch(
            plan,
            time_text
        )

        medication_mismatches = check_medication_mismatch(
            plan,
            time_text
        )

        # V6 schema (starts from the config the frontend holds,
        # so the user can add / change / remove from the chat)

        config, invalid, missing, summary, notes = (
            convert_to_new_schema(
                plan,
                data.get("current_config")
            )
        )

        # Something needs a time (or similar) -> ask, save nothing.
        # The frontend sends "pending" back with the next message.

        if missing:

            for item in missing:
                logs.append(
                    f"❓ Need more info ({item['task']}): "
                    f"{item['question']}"
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
                "reply": build_question_reply(missing),
                "needs_input": True,
                "missing": missing,
                "invalid": time_parse_invalid + invalid_days,
                "pending": {"text": user_text},
                "logs": logs
            })

        invalid = (
            time_parse_invalid
            + invalid_days
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

        extras = {}

        goal = extract_hydration_goal(user_text)

        if goal and any(
            isinstance(t, dict)
            and t.get("type") == "hydration"
            and normalize_action(t.get("action")) == "set"
            for t in parsed.get("tasks") or []
        ):

            hydration_name = DISPLAY_NAMES["hydration"]

            for i, line in enumerate(summary):
                if line.startswith(hydration_name):
                    summary[i] = f"{line} · {goal}"
                    extras["hydration_goal"] = goal

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
                "bottle_clean",
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
                "Custom habit count: "
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

            "summary":
                summary,

            "notes":
                notes,

            "extras":
                extras,

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
