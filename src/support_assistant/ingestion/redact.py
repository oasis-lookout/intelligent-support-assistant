"""PII redaction applied at ingestion time AND to user messages before they are stored.

Deliberately conservative: it keeps "ending 4417"-style last-4 references and INR amounts
(with separators) because they carry meaning for support answers.
"""
from __future__ import annotations

import re

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_AADHAAR = re.compile(r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_LONGNUM = re.compile(r"(?<![\d,])\d{9,}(?![\d,])")


def redact(text: str) -> str:
    text = _EMAIL.sub("[EMAIL]", text)
    text = _PAN.sub("[PAN]", text)
    text = _AADHAAR.sub("[ID_NUMBER]", text)
    text = _PHONE.sub("[PHONE]", text)
    text = _CARD.sub("[NUMBER]", text)
    text = _LONGNUM.sub("[NUMBER]", text)
    return text
