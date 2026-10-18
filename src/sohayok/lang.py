"""Language helpers: tell Bangla from English, normalize text, and read Bangla numbers."""

from __future__ import annotations

import re
import unicodedata

BANGLA_RANGE = ("ঀ", "৿")
BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_WORD = re.compile(r"[ঀ-৿]+|[A-Za-z]+")

# Bangla number words up to 100 that people say in commands ("ভলিউম পঞ্চাশ করো").
_BN_NUMBERS = {
    "শূন্য": 0,
    "এক": 1,
    "দুই": 2,
    "দু": 2,
    "তিন": 3,
    "চার": 4,
    "পাঁচ": 5,
    "ছয়": 6,
    "সাত": 7,
    "আট": 8,
    "নয়": 9,
    "দশ": 10,
    "এগারো": 11,
    "বারো": 12,
    "তেরো": 13,
    "চৌদ্দ": 14,
    "পনেরো": 15,
    "ষোলো": 16,
    "সতেরো": 17,
    "আঠারো": 18,
    "উনিশ": 19,
    "বিশ": 20,
    "পঁচিশ": 25,
    "ত্রিশ": 30,
    "পঁয়ত্রিশ": 35,
    "চল্লিশ": 40,
    "পঁয়তাল্লিশ": 45,
    "পঞ্চাশ": 50,
    "পঞ্চান্ন": 55,
    "ষাট": 60,
    "পঁয়ষট্টি": 65,
    "সত্তর": 70,
    "পঁচাত্তর": 75,
    "আশি": 80,
    "পঁচাশি": 85,
    "নব্বই": 90,
    "পঁচানব্বই": 95,
    "একশো": 100,
    "একশ": 100,
    "একশত": 100,
}
_EN_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "fifteen": 15,
    "twenty": 20,
    "twenty five": 25,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "a hundred": 100,
    "one hundred": 100,
}


def normalize(text: str) -> str:
    """NFC-normalize, convert Bangla digits, collapse whitespace, and lowercase Latin."""
    text = unicodedata.normalize("NFC", text).translate(BANGLA_DIGITS)
    text = text.replace("‌", "").replace("‍", "")
    return re.sub(r"\s+", " ", text).strip().lower()


def strip_punctuation(text: str) -> str:
    r"""Replace punctuation and symbols (incl. the dari ।) with spaces.

    Regex \w is not usable for this: Bangla vowel signs and the hasanta are combining
    marks, not word characters, so [^\w] would cut every Bangla word apart.
    """
    out = [" " if unicodedata.category(c)[0] in "PS" else c for c in text]
    return re.sub(r"\s+", " ", "".join(out)).strip()


def bangla_share(text: str) -> float:
    """Fraction of words written in Bangla script (0 when there are no words)."""
    words = _WORD.findall(text)
    if not words:
        return 0.0
    bangla = sum(1 for word in words if BANGLA_RANGE[0] <= word[0] <= BANGLA_RANGE[1])
    return bangla / len(words)


def detect_language(text: str, default: str = "en") -> str:
    """Return "bn" when most words are Bangla script, "en" for Latin, else the default.

    Romanized Bangla (Banglish) is Latin script, so it is reported as "en" here; the
    speech recognizer's language decision is what picks the reply voice for speech.
    """
    words = _WORD.findall(text)
    if not words:
        return default
    return "bn" if bangla_share(text) >= 0.5 else "en"


def parse_number(text: str) -> int | None:
    """Read the first number in a command, from digits or Bangla/English number words."""
    text = normalize(text)
    match = re.search(r"\d+", text)
    if match:
        return int(match.group())
    for word, value in _BN_WORDS:
        if re.search(rf"(?<![ঀ-৿]){word}(?![ঀ-৿])", text):
            return value
    for word, value in _EN_WORDS:
        if re.search(rf"\b{word}\b", text):
            return value
    return None


# Longest words first, so "পঁচিশ" is not read as a shorter word inside it.
_BN_WORDS = sorted(
    {normalize(k): v for k, v in _BN_NUMBERS.items()}.items(), key=lambda kv: -len(kv[0])
)
_EN_WORDS = sorted(_EN_NUMBERS.items(), key=lambda kv: -len(kv[0]))


def to_bangla_digits(value: int | str) -> str:
    return str(value).translate(str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯"))
