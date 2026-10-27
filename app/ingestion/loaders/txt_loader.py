"""Plain-text loader with encoding fallback."""

from __future__ import annotations

import re
from pathlib import Path

from .base import Section, clean_text

_ENCODINGS = ("utf-8-sig", "utf-16", "cp1252")


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in _ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")  # never fails


# Plain-text heading conventions: a title underlined with ===, or a line framed by --- rules.
_FRAMED_HEADING = re.compile(r"^-{3,}[ \t]*\n(?P<title>[^\n]+)\n-{3,}[ \t]*$", re.MULTILINE)
_UNDERLINED_H1 = re.compile(r"^(?P<title>[^\n]*\S[^\n]*)\n={3,}[ \t]*$", re.MULTILINE)
_UNDERLINED_H2 = re.compile(r"^(?P<title>[^\n]*[A-Za-z][^\n]*)\n-{3,}[ \t]*$", re.MULTILINE)


def markdown_headings(text: str) -> str:
    """Rewrite underlined / rule-framed headings as Markdown so the chunker can track sections."""
    text = _FRAMED_HEADING.sub(lambda m: f"\n## {m['title'].strip()}\n", text)
    text = _UNDERLINED_H1.sub(lambda m: f"# {m['title'].strip()}", text)
    text = _UNDERLINED_H2.sub(lambda m: f"## {m['title'].strip()}", text)
    return text


def load_txt(path: Path) -> list[Section]:
    text = clean_text(markdown_headings(read_text(path)))
    return [Section(text=text)] if text else []
