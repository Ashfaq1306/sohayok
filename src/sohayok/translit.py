"""Rough Bangla-to-Latin transliteration, for matching spoken names to saved contacts.

"রহিম" -> "rohim", "আশফাক" -> "ashfak": close enough for fuzzy matching against "Rahim" or
"Ashfaq" (contacts are often saved in English letters while the recognizer writes Bangla).
"""

from __future__ import annotations

import unicodedata

CONSONANTS = {
    "ক": "k",
    "খ": "kh",
    "গ": "g",
    "ঘ": "gh",
    "ঙ": "ng",
    "চ": "ch",
    "ছ": "chh",
    "জ": "j",
    "ঝ": "jh",
    "ঞ": "n",
    "ট": "t",
    "ঠ": "th",
    "ড": "d",
    "ঢ": "dh",
    "ণ": "n",
    "ত": "t",
    "থ": "th",
    "দ": "d",
    "ধ": "dh",
    "ন": "n",
    "প": "p",
    "ফ": "f",
    "ব": "b",
    "ভ": "bh",
    "ম": "m",
    "য": "j",
    "র": "r",
    "ল": "l",
    "শ": "sh",
    "ষ": "sh",
    "স": "s",
    "হ": "h",
    "ড়": "r",
    "ঢ়": "rh",
    "য়": "y",
    "ৎ": "t",
}
VOWELS = {
    "অ": "o",
    "আ": "a",
    "ই": "i",
    "ঈ": "i",
    "উ": "u",
    "ঊ": "u",
    "ঋ": "ri",
    "এ": "e",
    "ঐ": "oi",
    "ও": "o",
    "ঔ": "ou",
}
SIGNS = {
    "া": "a",
    "ি": "i",
    "ী": "i",
    "ু": "u",
    "ূ": "u",
    "ৃ": "ri",
    "ে": "e",
    "ৈ": "oi",
    "ো": "o",
    "ৌ": "ou",
}
VIRAMA, NUKTA = "্", "়"
MARKS = {"ং": "ng", "ঃ": "h", "ঁ": "n"}


NUKTA_LETTERS = {"ড": "r", "ঢ": "rh", "য": "y"}  # ড়, ঢ় and য় are letter + nukta after NFC


def to_latin(text: str) -> str:
    chars = list(unicodedata.normalize("NFC", text))
    out: list[str] = []
    for i, ch in enumerate(chars):
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if ch == "য" and i > 0 and chars[i - 1] == VIRAMA and nxt != NUKTA:
            continue  # য-ফলা ("ট্যা") is a glide, not a "j"
        if ch in NUKTA_LETTERS and nxt == NUKTA:
            sound, nxt = NUKTA_LETTERS[ch], chars[i + 2] if i + 2 < len(chars) else ""
        elif ch in CONSONANTS:
            sound = CONSONANTS[ch]
        else:
            sound = None
        if sound is not None:
            out.append(sound)
            # inherent vowel "o", unless a vowel sign or virama follows, or the word ends
            if nxt and nxt not in SIGNS and nxt not in (VIRAMA, NUKTA) and nxt not in MARKS:
                after_vowel = i > 0 and (chars[i - 1] in SIGNS or chars[i - 1] in VOWELS)
                if nxt.strip() and not (nxt in CONSONANTS and after_vowel):
                    out.append("o")  # "রহিম": no vowel between হি and ম; "আশফাক": none after আশ
        elif ch in VOWELS:
            out.append(VOWELS[ch])
        elif ch in SIGNS:
            out.append(SIGNS[ch])
        elif ch in MARKS:
            out.append(MARKS[ch])
        elif ch in (VIRAMA, NUKTA, "‌", "‍"):
            continue
        elif ch == "য" and i > 0 and chars[i - 1] == VIRAMA:
            continue  # য-ফলা ("ট্যা") is a glide, not a "j"
        else:
            out.append(ch)
    return "".join(out)
