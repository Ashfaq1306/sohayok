"""Timers and reminders, kept across restarts, plus the weather."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import httpx

from sohayok.tools.registry import ToolError, number, string, tool


@dataclass
class Reminder:
    id: str
    due: float  # Unix time
    text: str


class Scheduler:
    """Fires reminders on a background thread and persists them to a JSON file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.items: dict[str, Reminder] = {}
        self.on_fire: Callable[[Reminder], None] = lambda r: None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        if path:
            self.load(path)

    def load(self, path: Path) -> None:
        """Use `path` for persistence, restoring reminders saved there."""
        self.path = path
        if path.exists():
            with self._lock:
                for raw in json.loads(path.read_text(encoding="utf-8")):
                    self.items[raw["id"]] = Reminder(**raw)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="reminders", daemon=True)
            self._thread.start()

    def add(self, due: float, text: str) -> Reminder:
        reminder = Reminder(uuid.uuid4().hex[:8], due, text)
        with self._lock:
            self.items[reminder.id] = reminder
            self._save()
        self._wake.set()
        return reminder

    def cancel(self, reminder_id: str) -> bool:
        with self._lock:
            found = self.items.pop(reminder_id, None) is not None
            self._save()
        return found

    def pending(self) -> list[Reminder]:
        with self._lock:
            return sorted(self.items.values(), key=lambda r: r.due)

    def due_now(self, now: float | None = None) -> list[Reminder]:
        now = time.time() if now is None else now
        with self._lock:
            due = [r for r in self.items.values() if r.due <= now]
            for reminder in due:
                del self.items[reminder.id]
            if due:
                self._save()
        return due

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps([asdict(r) for r in self.items.values()], ensure_ascii=False),
                encoding="utf-8",
            )

    def _run(self) -> None:
        while True:
            for reminder in self.due_now():
                try:
                    self.on_fire(reminder)
                except Exception:  # an announcement failure must not stop the scheduler
                    pass
            upcoming = self.pending()
            delay = min(30.0, max(0.2, upcoming[0].due - time.time())) if upcoming else 30.0
            self._wake.wait(delay)
            self._wake.clear()


SCHEDULER = Scheduler()


def parse_clock(text: str, now: datetime | None = None) -> datetime:
    """Parse '7:30', '19:30', '7 pm', '7:30 am' into the next such moment."""
    now = now or datetime.now()
    match = re.fullmatch(r"\s*(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)?\s*", text.lower())
    if not match:
        raise ToolError(f"cannot read the time {text!r}; use HH:MM")
    hour, minute = int(match.group(1)), int(match.group(2) or 0)
    if match.group(3):
        hour = hour % 12 + (12 if match.group(3).startswith("p") else 0)
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ToolError(f"invalid time {text!r}")
    due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return due if due > now else due + timedelta(days=1)


@tool(
    "Set a timer, or a reminder at a clock time. Give either minutes (and/or seconds) from "
    "now, or at='HH:MM' (24-hour or with am/pm). The text is spoken when it fires.",
    {
        "text": string("What to remind about, e.g. 'tea is ready' / 'নামাজের সময়'"),
        "minutes": number("Minutes from now"),
        "seconds": number("Seconds from now"),
        "at": string("Clock time, e.g. '18:30' or '6:30 pm'"),
    },
    required=["text"],
)
def set_reminder(text: str, minutes: float = 0, seconds: float = 0, at: str = "") -> str:
    if at:
        due = parse_clock(at)
    elif minutes or seconds:
        due = datetime.now() + timedelta(minutes=float(minutes), seconds=float(seconds))
    else:
        raise ToolError("say when: minutes from now or a clock time")
    reminder = SCHEDULER.add(due.timestamp(), text)
    return f"reminder {reminder.id} set for {due:%H:%M:%S} ({due:%A})"


@tool("List pending timers and reminders.")
def list_reminders() -> list[dict[str, str]]:
    return [
        {"id": r.id, "at": datetime.fromtimestamp(r.due).strftime("%a %H:%M:%S"), "text": r.text}
        for r in SCHEDULER.pending()
    ]


@tool("Cancel a timer or reminder by id (or 'all').", {"reminder_id": string("Id or 'all'")})
def cancel_reminder(reminder_id: str) -> str:
    if reminder_id == "all":
        count = 0
        for reminder in SCHEDULER.pending():
            count += SCHEDULER.cancel(reminder.id)
        return f"cancelled {count}"
    if not SCHEDULER.cancel(reminder_id):
        raise ToolError(f"no reminder {reminder_id!r}")
    return "cancelled"


# Bangla names of Bangladesh's divisions, districts, and other common places; the weather
# service only searches English names.
BANGLA_PLACES = {
    "ঢাকা": "Dhaka",
    "চট্টগ্রাম": "Chittagong",
    "চট্রগ্রাম": "Chittagong",
    "সিলেট": "Sylhet",
    "রাজশাহী": "Rajshahi",
    "খুলনা": "Khulna",
    "বরিশাল": "Barisal",
    "রংপুর": "Rangpur",
    "ময়মনসিংহ": "Mymensingh",
    "কুমিল্লা": "Comilla",
    "গাজীপুর": "Gazipur",
    "নারায়ণগঞ্জ": "Narayanganj",
    "কক্সবাজার": "Cox's Bazar",
    "বগুড়া": "Bogra",
    "যশোর": "Jessore",
    "দিনাজপুর": "Dinajpur",
    "ফরিদপুর": "Faridpur",
    "নোয়াখালী": "Noakhali",
    "ফেনী": "Feni",
    "টাঙ্গাইল": "Tangail",
    "পাবনা": "Pabna",
    "কুষ্টিয়া": "Kushtia",
    "সাভার": "Savar",
    "নরসিংদী": "Narsingdi",
    "ব্রাহ্মণবাড়িয়া": "Brahmanbaria",
    "চাঁদপুর": "Chandpur",
    "লক্ষ্মীপুর": "Lakshmipur",
    "পটুয়াখালী": "Patuakhali",
    "ভোলা": "Bhola",
    "সাতক্ষীরা": "Satkhira",
    "বাগেরহাট": "Bagerhat",
    "নাটোর": "Natore",
    "নওগাঁ": "Naogaon",
    "জামালপুর": "Jamalpur",
    "শেরপুর": "Sherpur",
    "নেত্রকোনা": "Netrokona",
    "কিশোরগঞ্জ": "Kishoreganj",
    "হবিগঞ্জ": "Habiganj",
    "মৌলভীবাজার": "Moulvibazar",
    "সুনামগঞ্জ": "Sunamganj",
    "রাঙ্গামাটি": "Rangamati",
    "বান্দরবান": "Bandarban",
    "খাগড়াছড়ি": "Khagrachhari",
    "কুড়িগ্রাম": "Kurigram",
    "গাইবান্ধা": "Gaibandha",
    "ঠাকুরগাঁও": "Thakurgaon",
    "পঞ্চগড়": "Panchagarh",
    "নীলফামারী": "Nilphamari",
    "লালমনিরহাট": "Lalmonirhat",
    "জয়পুরহাট": "Joypurhat",
    "সিরাজগঞ্জ": "Sirajganj",
    "মানিকগঞ্জ": "Manikganj",
    "মুন্সীগঞ্জ": "Munshiganj",
    "গোপালগঞ্জ": "Gopalganj",
    "মাদারীপুর": "Madaripur",
    "শরীয়তপুর": "Shariatpur",
    "রাজবাড়ী": "Rajbari",
    "মাগুরা": "Magura",
    "ঝিনাইদহ": "Jhenaidah",
    "নড়াইল": "Narail",
    "চুয়াডাঙ্গা": "Chuadanga",
    "মেহেরপুর": "Meherpur",
    "পিরোজপুর": "Pirojpur",
    "ঝালকাঠি": "Jhalokati",
    "বরগুনা": "Barguna",
    "চাঁপাইনবাবগঞ্জ": "Chapai Nawabganj",
    "সেন্টমার্টিন": "Saint Martin's Island",
    "কলকাতা": "Kolkata",
    "দিল্লি": "Delhi",
    "লন্ডন": "London",
    "নিউ ইয়র্ক": "New York",
    "দুবাই": "Dubai",
    "সৌদি আরব": "Riyadh",
    "মক্কা": "Mecca",
    "টোকিও": "Tokyo",
}


def _normalized_places() -> dict[str, str]:
    from sohayok.lang import normalize

    return {normalize(k): v for k, v in BANGLA_PLACES.items()}


def english_city(city: str) -> str:
    """'ঢাকার' / 'ঢাকা শহর' / 'Dhaka' -> a name the geocoder knows."""
    from sohayok.lang import normalize

    key = normalize(city).strip(" ?।")
    for suffix in ("শহরের", "শহর", "জেলার", "জেলা", "এর", "য়ের", "ের", "র", "তে", "য়", "এ", "ে"):
        if key in _PLACES:
            break
        if key.endswith(suffix):
            key = key[: -len(suffix)].strip()
    return _PLACES.get(key, city)


_PLACES = _normalized_places()


def bangla_genitive(english_name: str) -> str:
    """'Dhaka' -> 'ঢাকার', 'Sylhet' -> 'সিলেটের' (other places keep their English name)."""
    names: dict[str, str] = {}
    for bangla, english in BANGLA_PLACES.items():
        names.setdefault(english, bangla)  # the first (standard) spelling wins
    name = names.get(english_name)
    if name is None:
        return f"{english_name}-এর"
    return name + ("র" if name[-1] in "ািীুূেৈোৌ" else "ের")


def bn_number(value: float) -> str:
    from sohayok.lang import to_bangla_digits

    return to_bangla_digits(round(value))


WEATHER_BN = {
    0: "আকাশ পরিষ্কার",
    1: "আকাশ প্রায় পরিষ্কার",
    2: "আংশিক মেঘলা",
    3: "আকাশ মেঘলা",
    45: "কুয়াশা",
    48: "কুয়াশা",
    51: "হালকা গুঁড়ি গুঁড়ি বৃষ্টি",
    53: "গুঁড়ি গুঁড়ি বৃষ্টি",
    55: "ঘন গুঁড়ি বৃষ্টি",
    61: "হালকা বৃষ্টি",
    63: "বৃষ্টি",
    65: "ভারী বৃষ্টি",
    71: "হালকা তুষার",
    73: "তুষারপাত",
    75: "ভারী তুষারপাত",
    80: "দমকা বৃষ্টি",
    81: "দমকা বৃষ্টি",
    82: "প্রবল দমকা বৃষ্টি",
    95: "বজ্রঝড়",
    96: "শিলাবৃষ্টিসহ বজ্রঝড়",
    99: "শিলাবৃষ্টিসহ বজ্রঝড়",
}
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    80: "rain showers",
    81: "rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with hail",
}


@tool(
    "Current weather and today's forecast for a city (default: the configured home city).",
    {"city": string("City name, e.g. Dhaka, Chattogram, Sylhet")},
    required=[],
)
def weather(city: str = "") -> dict[str, str]:
    from sohayok.config import current

    city = english_city(city or current().assistant.home_city)
    with httpx.Client(timeout=10) as client:
        places = (
            client.get(
                "https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1}
            )
            .json()
            .get("results")
        )
        if not places:
            raise ToolError(f"unknown city {city!r}")
        place = places[0]
        data = client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 1,
            },
        ).json()
    now, daily = data["current"], data["daily"]
    code = now["weather_code"]
    return {
        "spoken_bn": (
            f"{bangla_genitive(place['name'])} আবহাওয়া: এখন {bn_number(now['temperature_2m'])} ডিগ্রি, "
            f"{WEATHER_BN.get(code, 'আবহাওয়া অজানা')}। আজ "
            f"{bn_number(daily['temperature_2m_min'][0])} থেকে "
            f"{bn_number(daily['temperature_2m_max'][0])} ডিগ্রি, বৃষ্টির সম্ভাবনা "
            f"{bn_number(daily['precipitation_probability_max'][0])} শতাংশ।"
        ),
        "spoken_en": (
            f"In {place['name']} it's {round(now['temperature_2m'])} degrees and "
            f"{WEATHER_CODES.get(code, 'unknown')}. Today {round(daily['temperature_2m_min'][0])} "
            f"to {round(daily['temperature_2m_max'][0])} degrees, "
            f"{daily['precipitation_probability_max'][0]}% chance of rain."
        ),
        "place": f"{place['name']}, {place.get('country', '')}",
        "now": f"{now['temperature_2m']}°C, {WEATHER_CODES.get(now['weather_code'], 'unknown')}, "
        f"humidity {now['relative_humidity_2m']}%, wind {now['wind_speed_10m']} km/h",
        "today": f"{daily['temperature_2m_min'][0]}–{daily['temperature_2m_max'][0]}°C, "
        f"rain chance {daily['precipitation_probability_max'][0]}%",
    }
