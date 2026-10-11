"""SMS as a modality: encoding, segments, plain text and carrier keywords."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

# GSM 03.38 basic character set and the extension table (2 septets each).
GSM_BASIC = set(
    "@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?"
    "¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà"
)
GSM_EXT = set("^{}\\[~]|€\f")


@dataclass(frozen=True)
class SegmentInfo:
    encoding: str  # "GSM-7" or "UCS-2"
    units: int  # septets for GSM-7, UTF-16 code units for UCS-2
    segments: int

    @property
    def label(self) -> str:
        noun = "segment" if self.segments == 1 else "segments"
        return f"{self.segments} {noun} · {self.encoding}"


def segment_info(text: str) -> SegmentInfo:
    """Count segments the way carriers do: 160/153 for GSM-7, 70/67 for UCS-2."""
    if all(c in GSM_BASIC or c in GSM_EXT for c in text):
        units = sum(2 if c in GSM_EXT else 1 for c in text)
        single, multi, enc = 160, 153, "GSM-7"
    else:
        units = len(text.encode("utf-16-le")) // 2
        single, multi, enc = 70, 67, "UCS-2"
    if units <= single:
        return SegmentInfo(enc, units, 1)
    return SegmentInfo(enc, units, -(-units // multi))


_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_EMPH = re.compile(r"(\*\*|__|\*|`)(.+?)\1")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_BULLET = re.compile(r"^\s*[-*•]\s+", re.M)
_SMART = str.maketrans(
    {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
    }
)


def to_plain_text(text: str) -> str:
    """Strip markdown and swap typographic characters that force UCS-2."""
    out = _LINK.sub(r"\1 (\2)", text)
    out = _EMPH.sub(r"\2", out)
    out = _HEADING.sub("", out)
    out = _BULLET.sub("", out)
    out = out.translate(_SMART)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


class Keyword(StrEnum):
    STOP = "stop"
    START = "start"
    HELP = "help"


STOP_WORDS = {"STOP", "STOPALL", "UNSUBSCRIBE", "CANCEL", "END", "QUIT", "OPTOUT"}
START_WORDS = {"START", "UNSTOP", "OPTIN"}
HELP_WORDS = {"HELP", "INFO"}

TEMPLATES: dict[Keyword, str] = {
    Keyword.STOP: (
        "Kettle & Coil Repair: you are unsubscribed and will get no more texts. Reply START to resubscribe."
    ),
    Keyword.START: "Kettle & Coil Repair: you are resubscribed. Reply HELP for help, STOP to opt out.",
    Keyword.HELP: (
        "Kettle & Coil Repair (synthetic demo): text us about a repair. "
        "Reply STOP to opt out. Msg rates may apply."
    ),
}


def keyword(body: str) -> Keyword | None:
    """Carrier keywords match the whole message, ignoring case and punctuation."""
    word = re.sub(r"[^A-Za-z]", "", body).upper()
    if word in STOP_WORDS:
        return Keyword.STOP
    if word in START_WORDS:
        return Keyword.START
    if word in HELP_WORDS:
        return Keyword.HELP
    return None
