from sohayok.lang import bangla_share, detect_language, normalize, parse_number, to_bangla_digits


def test_detects_bangla_english_and_mixed():
    assert detect_language("ক্রোম খোলো") == "bn"
    assert detect_language("open chrome") == "en"
    assert detect_language("Chrome খোলো আর YouTube চালাও") == "bn"
    assert detect_language("chrome kholo") == "en"  # Banglish is Latin script
    assert detect_language("123 !!", default="bn") == "bn"


def test_bangla_share_counts_words():
    assert bangla_share("আমি Chrome চাই") == 2 / 3
    assert bangla_share("") == 0.0


def test_normalize_digits_case_and_spaces():
    assert normalize("  ভলিউম   ৫০  করো ") == "ভলিউম 50 করো"
    assert normalize("Open   CHROME") == "open chrome"


def test_parse_number_from_digits_and_words():
    assert parse_number("ভলিউম ৭০ করো") == 70
    assert parse_number("ভলিউম পঞ্চাশ") == 50
    assert parse_number("পঁচিশ") == 25  # not read as a shorter word inside it
    assert parse_number("set volume to thirty") == 30
    assert parse_number("no number here") is None


def test_to_bangla_digits():
    assert to_bangla_digits(2026) == "২০২৬"
