"""Shared types for document loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class Section:
    """A unit of extracted text (a PDF page, a PPTX slide, or a whole document).

    Paragraphs inside ``text`` are separated by blank lines so the chunker can split on them.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Loader(Protocol):
    def __call__(self, path: Path) -> list[Section]: ...


def clean_text(text: str) -> str:
    """Normalize whitespace while keeping paragraph breaks (blank lines)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [line.rstrip() for line in text.split("\n")]
    out: list[str] = []
    blank = False
    for line in lines:
        if not line.strip():
            if not blank and out:
                out.append("")
            blank = True
        else:
            out.append(line)
            blank = False
    return "\n".join(out).strip()


def rows_to_markdown(rows: list[list[str]]) -> str:
    """Render table rows as a Markdown table (first row is the header)."""
    rows = [[" ".join(cell.split()).replace("|", "\\|") for cell in row] for row in rows if any(c.strip() for c in row)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)
