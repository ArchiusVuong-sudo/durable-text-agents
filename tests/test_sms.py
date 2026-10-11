from __future__ import annotations

import pytest

from durable_text.sms import Keyword, keyword, segment_info, to_plain_text


@pytest.mark.parametrize(
    ("text", "encoding", "segments"),
    [
        ("a" * 160, "GSM-7", 1),
        ("a" * 161, "GSM-7", 2),
        ("a" * 306, "GSM-7", 2),
        ("a" * 307, "GSM-7", 3),
        ("é" * 160, "GSM-7", 1),  # in the GSM basic set
        ("€" * 80, "GSM-7", 1),  # extension table: 2 septets each
        ("€" * 81, "GSM-7", 2),
        ("ł" * 70, "UCS-2", 1),
        ("ł" * 71, "UCS-2", 2),
        ("ł" * 134, "UCS-2", 2),
        ("ł" * 135, "UCS-2", 3),
    ],
)
def test_segments(text: str, encoding: str, segments: int) -> None:
    info = segment_info(text)
    assert (info.encoding, info.segments) == (encoding, segments)


def test_emoji_counts_utf16_units() -> None:
    info = segment_info("ok \U0001f600")
    assert info.encoding == "UCS-2"
    assert info.units == 5  # the emoji is a surrogate pair


def test_segment_label() -> None:
    assert segment_info("hi").label == "1 segment · GSM-7"
    assert segment_info("x" * 200).label == "2 segments · GSM-7"


def test_plain_text_strips_markdown_and_smart_punctuation() -> None:
    md = (
        "**Good news**: your visit is *held*.\n- Tue 2\u20134 pm\n# Note\n"
        "See [details](https://example.test) \u2014 thanks\u2019"
    )
    out = to_plain_text(md)
    assert (
        out
        == "Good news: your visit is held.\nTue 2-4 pm\nNote\nSee details (https://example.test) - thanks'"
    )
    assert segment_info(out).encoding == "GSM-7"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("STOP", Keyword.STOP),
        ("stop", Keyword.STOP),
        (" Stop. ", Keyword.STOP),
        ("unsubscribe", Keyword.STOP),
        ("START", Keyword.START),
        ("help", Keyword.HELP),
        ("INFO", Keyword.HELP),
        ("yes", None),  # a normal reply, not a carrier keyword
        ("please stop texting me at night", None),
        ("helpful, thanks", None),
    ],
)
def test_keywords_match_whole_message(body: str, expected: Keyword | None) -> None:
    assert keyword(body) == expected
