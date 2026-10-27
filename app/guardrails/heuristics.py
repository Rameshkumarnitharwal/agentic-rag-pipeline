"""Cheap deterministic pre-checks that run before the LLM-based NeMo rails.

They catch obvious jailbreak/injection patterns without spending an LLM call. They are a first
line of defence only — anything they miss still goes through the NeMo self-check rails.
"""

from __future__ import annotations

import re

_INJECTION_PATTERNS = [
    r"\bignore\b.{0,40}\b(previous|prior|above|earlier|all|your)\b.{0,20}\b(instructions?|rules|prompts?|directions)\b",
    r"\bdisregard\b.{0,40}\b(instructions?|rules|prompts?|guidelines)\b",
    r"\bforget\b.{0,30}\b(your|all|previous)\b.{0,20}\b(instructions?|rules|training)\b",
    r"\b(reveal|print|show|repeat|output|leak)\b.{0,30}\b(system|hidden|initial|original)\s+(prompt|instructions?|message)\b",
    r"\byou are now\b.{0,40}\b(DAN|jailbroken|unrestricted|unfiltered|evil)\b",
    r"\bdo anything now\b",
    r"\b(developer|god|jailbreak|DAN)\s+mode\b",
    r"\bact as\b.{0,30}\b(no|without)\s+(restrictions|filters|rules|limits)\b",
    r"\bpretend\b.{0,40}\b(no|without)\s+(restrictions|filters|rules|guidelines)\b",
    r"^\s*(system|assistant|developer)\s*:",
    r"<\s*/?\s*(system|im_start|im_end)\s*>|\[/?INST\]|<\|im_start\|>",
    r"\bnew instructions?\s*:",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE | re.DOTALL) for p in _INJECTION_PATTERNS]

_SECRET_PATTERNS = [
    r"\bsk-[A-Za-z0-9_-]{20,}",  # OpenAI-style keys
    r"\bAKIA[0-9A-Z]{16}\b",  # AWS access key id
    r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"\bjina_[A-Za-z0-9]{20,}",
]
_SECRET_RE = [re.compile(p) for p in _SECRET_PATTERNS]


def detect_injection(text: str) -> str | None:
    """Return the matched snippet if the text looks like a jailbreak/prompt-injection attempt."""
    for pattern in _INJECTION_RE:
        match = pattern.search(text)
        if match:
            return match.group(0)[:80]
    return None


def detect_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SECRET_RE)
