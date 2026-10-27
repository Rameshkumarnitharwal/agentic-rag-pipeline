"""Deterministic PII detection and redaction for answers (and anything else that leaves the system).

Checksums (Luhn, Verhoeff) and format rules keep false positives low on technical text, where long
numbers (job ids, timestamps) are common. Documentation-reserved email domains are not redacted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --- checksums ---------------------------------------------------------------------------------

_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5], [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7], [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3], [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4], [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7], [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(number: str) -> bool:
    c = 0
    for i, digit in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(digit)]]
    return c == 0


def luhn_valid(number: str) -> bool:
    total = 0
    for i, digit in enumerate(reversed(number)):
        d = int(digit)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _card_brand_ok(digits: str) -> bool:
    n = len(digits)
    if digits[0] == "4":
        return n in (13, 16, 19)  # Visa
    if digits[:2] in {"51", "52", "53", "54", "55"} or 2221 <= int(digits[:4]) <= 2720:
        return n == 16  # Mastercard
    if digits[:2] in {"34", "37"}:
        return n == 15  # Amex
    if digits.startswith(("6011", "65")) or digits[:3] in {"644", "645", "646", "647", "648", "649"}:
        return n in (16, 19)  # Discover
    if digits[:2] in {"60", "65", "81", "82"}:
        return n == 16  # RuPay
    return False


# --- detectors ---------------------------------------------------------------------------------

_DOC_EMAIL_DOMAINS = ("example.com", "example.org", "example.net", "example.edu", "test.com", "domain.com")

_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    "aadhaar": re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{3}[ABCFGHJLPT][A-Z]\d{4}[A-Z]\b"),
    "phone": re.compile(r"(?<![\w+])(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}\b|\+(?:[1-9]\d{0,2})[ -]?\(?\d{2,4}\)?[ -]?\d{3,4}[ -]?\d{3,4}\b"),
    "ssn": re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    "ip_address": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
}

DEFAULT_ENTITIES = ("email", "credit_card", "aadhaar", "pan", "phone", "ssn")  # ip_address is opt-in


def _valid(entity: str, value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if entity == "email":
        return not value.lower().endswith(_DOC_EMAIL_DOMAINS)
    if entity == "credit_card":
        return 13 <= len(digits) <= 19 and _card_brand_ok(digits) and luhn_valid(digits)
    if entity == "aadhaar":
        return len(digits) == 12 and verhoeff_valid(digits)
    return True


@dataclass
class PIIMatch:
    entity: str
    start: int
    end: int


def detect_pii(text: str, entities: tuple[str, ...] | list[str] = DEFAULT_ENTITIES) -> list[PIIMatch]:
    matches: list[PIIMatch] = []
    for entity in entities:
        for m in _PATTERNS[entity].finditer(text):
            if _valid(entity, m.group(0)):
                matches.append(PIIMatch(entity, m.start(), m.end()))
    # Resolve overlaps: keep the longest match (e.g. a card number over a phone-shaped substring).
    matches.sort(key=lambda m: (m.start, -(m.end - m.start)))
    kept: list[PIIMatch] = []
    for m in matches:
        if not kept or m.start >= kept[-1].end:
            kept.append(m)
    return kept


def redact_pii(text: str, entities: tuple[str, ...] | list[str] = DEFAULT_ENTITIES) -> tuple[str, list[str]]:
    """Replace PII with ``[REDACTED_<ENTITY>]``; returns the new text and the entity types found."""
    found = detect_pii(text, entities)
    for m in reversed(found):
        text = f"{text[:m.start]}[REDACTED_{m.entity.upper()}]{text[m.end:]}"
    return text, [m.entity for m in found]
