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


# Bangla words that English never produces when the Bangla recognizer writes English speech
# phonetically ("please open this PC" -> "প্লিজ ওপেন ডিস পিসি"): pronouns, question words,
# particles, and everyday verb forms. Words that English also produces ("a" -> এ, "are" -> আর,
# "they" -> দে, "then" -> দেন, "ball" -> বল, ".com" -> কম) are left out.
_BANGLA_GRAMMAR = {
    unicodedata.normalize("NFC", word)
    for word in """
 আমি আমার আমাকে আমাদের আমরা তুমি তোমার তোমাকে তুই তোর আপনি আপনার আপনাকে তার তাকে
 ওকে তারা তাদের এর একে এটা এটি ওটা ওটি সেটা সেটি এই ওই সেই এগুলো ওগুলো কেউ কিছু
 কি কী কেন কোথায় কোথা কখন কেমন কত কতো কোন কোনটা কে কার কাকে কিভাবে কীভাবে
 এবং কিন্তু না নাই নেই নয় হ্যাঁ হা জি জ্বি তো যে যদি তাহলে তবে শুধু আবার আরো আরও একটু
 একটা একটি দুইটা দুটো সব সবাই খুব অনেক বেশি এখন এখনই আজ আজকে কাল পরে আগে এখানে ওখানে
 সেখানে ঐখানে উপরে নিচে ভিতরে বাইরে থেকে দিয়ে জন্য মধ্যে সাথে সঙ্গে কাছে মতো মত পর্যন্ত
 করো কর করেন করুন করে করছি করছে করবা করবে করবেন করতে করেছি করেছে করলাম করলে করি করবো করব
 খোলো খোল খুলো খুলে খুলুন খুলবা খুলবে বন্ধ দাও দেও দিন দিবা দিবে দিয়ো যাও যা যান যাবে
 চলো চল চালাও চালু চালিয়ে বাড়াও বাড়ান কমাও কমান বলো বলেন বলুন বলবা লেখো লেখ লিখো লিখ
 লেখবা লিখবা লিখে দেখাও দেখো দেখ দেখি দেখবা পাঠাও পাঠান পাঠিয়ে ফেলো ফেল নাও নিন রাখো রাখ
 থামো থাম থামাও ছাড়ো ছাড় ছাড়া শোনো শোন শুনি চাই চাচ্ছি চাইছি পারো পারবে পারবা পারি হবে হচ্ছে
 হয়েছে হলো আছে আছো আছেন ছিল ছিলো নেই লাগবে লাগে দরকার মানে ঠিক আচ্ছা ভালো ধন্যবাদ

    দেখুন শুনুন চালান চালাও বন্ধ নিয়ে নিয়ো নিবা নিবে হয় হয়নি হয়েছিল হয়েছিলো হবে না আসো আসছি এসো
    আসবে যাবো যাব যাচ্ছি পারছি পারছো পারছেন লাগছে দিচ্ছি দিচ্ছে দিয়েছি করছো করছেন করছিলাম
    গতকাল আগামীকাল সকালে বিকেলে রাতে দুপুরে কালকে মধ্যেই এখানেই ভাই আপু আম্মু আব্বু
    """.split()
}
# Bangla endings on English loanwords: "ইউটিউবের", "ট্যাবটা", "ফাইলগুলো", "ক্রোমে".
_BANGLA_SUFFIXES = tuple(
    unicodedata.normalize("NFC", s)
    for s in ("ের", "য়ের", "টা", "টি", "টাকে", "টাতে", "গুলো", "গুলা", "গুলি", "কে", "তে", "য়ে")
)
_NOT_SUFFIXED = {
    unicodedata.normalize("NFC", w) for w in ("প্লে", "টুডে", "পকেট", "লাইক", "ব্ল্যাকে", "হোয়াইটে")
}


# "WhatsApp" as the recognizers write it: হোয়াটসঅ্যাপ, হুয়াটস্যাপ, ওয়াটস্যাপ, হোয়াটসআপ, …
WHATSAPP = r"(?:whats ?app|[হও][ো ু]?য়াটস ?(?:্যা|অ্যা|আ|এ|য়া|্য)প|হটস ?অ্যাপ)"


def _letters(word: str) -> int:
    """Letters, not code points: vowel signs, the virama and the nukta do not count."""
    return sum(unicodedata.category(ch) == "Lo" for ch in word)


def has_bangla_grammar(text: str, short_is_bangla: bool = True) -> bool:
    """Does Bangla-script text contain Bangla grammar, or only English written in Bangla letters?

    The Bangla recognizer handles the user's mixed speech ("ইউটিউবের ট্যাবটা কেটে ফেলো") far
    better than English Whisper, but writes plain English speech phonetically; this tells the
    two apart.
    """
    words = [w for w in _WORD.findall(normalize(text)) if w[0] >= BANGLA_RANGE[0]]
    if not words:
        return False
    for word in words:
        if word in _BANGLA_GRAMMAR:
            return True
        # a real word before the ending: "ইউটিউবের" yes, "ওয়ের" ("where") no
        stems = [word[: -len(end)] for end in _BANGLA_SUFFIXES if word.endswith(end)]
        if word not in _NOT_SUFFIXED and any(_letters(stem) >= 3 for stem in stems):
            return True
    # one or two words with no grammar ("ক্রোম", "ইয়েস") cannot be told apart by grammar
    return short_is_bangla and len(words) <= 2
