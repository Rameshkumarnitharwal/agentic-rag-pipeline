"""Paragraph-based text splitter.

Paragraphs (blank-line separated) are packed greedily into chunks of at most ``max_chars``.
Markdown headings are tracked so every chunk is prefixed with its section path, and a new chunk
starts at a heading once the current one is reasonably full. A paragraph that is too large on its
own is split further — tables by rows (header repeated), fenced code by lines (re-fenced), other
text by sentences, then lines, then a hard cut as a last resort.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from app.ingestion.loaders.base import Section

DEFAULT_MAX_CHARS = 1500
DEFAULT_MIN_CHARS = 50

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_TABLE_DIVIDER = re.compile(r"^\|( --- \|)+$", re.MULTILINE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _is_table(paragraph: str) -> bool:
    lines = paragraph.splitlines()
    return len(lines) >= 2 and lines[0].startswith("|") and lines[1].startswith("|")


def _pack(pieces: list[str], max_chars: int, sep: str) -> list[str]:
    """Greedily join pieces with ``sep`` without exceeding ``max_chars``."""
    out: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{sep}{piece}" if current else piece
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                out.append(current)
            current = piece
    if current:
        out.append(current)
    return out


def _hard_split(text: str, max_chars: int) -> list[str]:
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _split_table(table: str, max_chars: int) -> list[str]:
    header, divider, *rows = table.splitlines()
    head = f"{header}\n{divider}"
    budget = max_chars - len(head) - 1
    if budget <= 0:
        return _hard_split(table, max_chars)
    rows = [r for row in rows for r in (_hard_split(row, budget) if len(row) > budget else [row])]
    return [f"{head}\n{body}" for body in _pack(rows, budget, "\n")]


CODE_ANCHOR_LINES = 3  # leading lines of a split code block repeated in each continuation


def _split_code(block: str, max_chars: int, lead_in: str = "") -> list[str]:
    """Split an oversized fenced code block by lines, re-fencing every piece.

    Continuation pieces repeat the block's lead-in sentence and its first few lines (e.g.
    ``kind: CronJob`` / ``name: database-backup``) so each piece still says what it belongs to.
    """
    lines = block.strip("`").strip("\n").splitlines()
    # Skip bare keys like "metadata:" — they say nothing about what the block is.
    anchor = [line for line in lines if line.strip() and not line.strip().endswith(":")][:CODE_ANCHOR_LINES]
    context = (f"(continued: {lead_in})\n" if lead_in else "(continued)\n") + "```\n" + "\n".join(anchor) + "\n# ...\n"
    if len(context) > max_chars // 3:  # anchor too big to be worth repeating
        context = "(continued)\n```\n"
    budget = max_chars - len(context) - 4  # room for context and the closing fence
    lines = [p for line in lines for p in (_hard_split(line, budget) if len(line) > budget else [line])]
    bodies = _pack(lines, budget, "\n")
    pieces = [f"```\n{bodies[0]}\n```"]
    for body in bodies[1:]:
        pieces.append(f"{context}{body}\n```")
    return pieces


def _split_oversized(paragraph: str, max_chars: int, lead_in: str = "") -> list[str]:
    if _is_table(paragraph):
        return _split_table(paragraph, max_chars)
    if paragraph.startswith("```"):
        return _split_code(paragraph, max_chars, lead_in)
    if paragraph.startswith("(continued"):  # a code continuation piece that needs splitting again
        first, _, rest = paragraph.partition("\n")
        if rest.startswith("```"):
            return _split_code(rest, max_chars, first.removeprefix("(continued").strip(" :)"))
    for splitter, sep in ((_SENTENCE_END, " "), (re.compile(r"\n"), "\n")):
        parts = [p for p in splitter.split(paragraph) if p.strip()]
        if len(parts) > 1:
            out: list[str] = []
            for part in _pack(parts, max_chars, sep):
                out.extend(_split_oversized(part, max_chars) if len(part) > max_chars else [part])
            return out
    return _hard_split(paragraph, max_chars)


_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
MAX_BREADCRUMB_CHARS = 200
# Start a new chunk at a heading once the current one is at least this full, so sections stay together
# without producing lots of tiny chunks.
SECTION_BREAK_FILL = 0.4


def _paragraphs(text: str) -> list[str]:
    """Blank-line separated paragraphs, keeping a fenced code block whole (and its indentation)."""
    out: list[str] = []
    pending: str | None = None
    for raw in _PARAGRAPH_BREAK.split(text):
        if pending is not None:
            pending += "\n\n" + raw.strip("\n").rstrip()
            if pending.count("```") % 2 == 0:
                out.append(pending)
                pending = None
            continue
        paragraph = raw.strip()
        if not paragraph:
            continue
        if paragraph.count("```") % 2 == 1:
            pending = paragraph
        else:
            out.append(paragraph)
    if pending is not None:
        out.append(pending)
    return out


def _heading(paragraph: str) -> tuple[int, str] | None:
    if "\n" in paragraph:
        return None
    match = _HEADING.match(paragraph)
    return (len(match.group(1)), match.group(2).strip()) if match else None


def _breadcrumb(path: list[tuple[int, str]], max_chars: int) -> str:
    if not path:
        return ""
    limit = min(MAX_BREADCRUMB_CHARS, max_chars // 5)  # never let the prefix crowd out content
    crumb = " > ".join(title for _, title in path)
    if len(crumb) > limit:
        crumb = "… > " + crumb[-(limit - 4):]
    return f"Section: {crumb}\n\n"


def split_text(text: str, max_chars: int = DEFAULT_MAX_CHARS, section_prefix: bool = True) -> list[str]:
    """Split text into paragraph-aligned chunks of at most ``max_chars`` characters.

    Markdown headings are tracked; with ``section_prefix`` each chunk starts with a
    ``Section: A > B > C`` line naming where it sits, so retrieval and the LLM see the context
    even when the heading itself is in an earlier chunk.
    """
    path: list[tuple[int, str]] = []  # current heading stack: (level, title)
    chunks: list[str] = []
    current: list[str] = []
    is_heading: list[bool] = []
    prefix = ""

    def length(extra: str | None = None) -> int:
        parts = current + ([extra] if extra is not None else [])
        return len(prefix) + sum(map(len, parts)) + 2 * max(len(parts) - 1, 0)

    def open_chunk(first: str, first_is_heading: bool) -> None:
        nonlocal prefix
        if section_prefix:
            # A chunk opening with a heading gets its parents' path; otherwise the full path.
            level = _heading(first)[0] if first_is_heading else 7
            prefix = _breadcrumb([h for h in path if h[0] < level], max_chars)
        else:
            prefix = ""

    def flush(carry_headings: bool = True) -> None:
        nonlocal current, is_heading
        carry: list[str] = []
        while carry_headings and current and is_heading[-1]:  # never end a chunk on a heading
            carry.insert(0, current.pop())
            is_heading.pop()
        if current:
            text = prefix + "\n\n".join(current)
            # Safety net for pathological input (e.g. very long headings): never exceed max_chars.
            chunks.extend(_split_oversized(text, max_chars) if len(text) > max_chars else [text])
        current, is_heading = [], []
        if carry:
            open_chunk(carry[0], True)
            current, is_heading = carry, [True] * len(carry)

    lead_in = ""  # last short prose line before a code block, e.g. "Complete CronJob Example:"

    def add(piece: str, heading: bool) -> None:
        if current and length(piece) > max_chars:
            flush()
        if not current:
            open_chunk(piece, heading)
        if length(piece) <= max_chars:
            current.append(piece)
            is_heading.append(heading)
            return
        # Too big even for a fresh chunk (or next to carried headings): split it to the space left.
        room = max_chars - length() - (2 if current else 0)
        if room < 100 and current:  # carried headings leave no useful room; emit them on their own
            flush(carry_headings=False)
            open_chunk(piece, heading)
            room = max_chars - len(prefix)
        # Later parts open new chunks, whose prefix can be the full section path: size them for that.
        fresh_room = max_chars - len(_breadcrumb(path, max_chars)) if section_prefix else max_chars
        parts = _split_oversized(piece, min(room, fresh_room), lead_in)
        current.append(parts[0])
        is_heading.append(False)
        for part in parts[1:]:
            add(part, False)

    for paragraph in _paragraphs(text):
        heading = _heading(paragraph)
        if heading:
            path = [h for h in path if h[0] < heading[0]] + [heading]
            lead_in = ""
            if any(not h for h in is_heading) and length() >= SECTION_BREAK_FILL * max_chars:
                flush()
        add(paragraph, heading is not None)
        if not heading and not paragraph.startswith(("```", "|")) and len(paragraph) <= 150:
            lead_in = " ".join(paragraph.split())
    flush(carry_headings=False)
    return chunks


def _chunk_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha256(f"{source}\x00{index}\x00{text}".encode()).hexdigest()
    return digest[:32]


def chunk_sections(
    sections: list[Section],
    base_metadata: dict[str, Any] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """Chunk each Section separately so page/slide metadata stays accurate.

    Chunks shorter than ``min_chars`` (page numbers, stray headers) are dropped.
    """
    base_metadata = base_metadata or {}
    source = str(base_metadata.get("source", ""))
    chunks: list[Chunk] = []
    for section in sections:
        for piece in split_text(section.text, max_chars):
            if len(piece) < min_chars:
                continue
            index = len(chunks)
            metadata = {
                **base_metadata,
                **section.metadata,
                "chunk_index": index,
                "char_count": len(piece),
                "content_type": "table" if _TABLE_DIVIDER.search(piece) else "text",
            }
            chunks.append(Chunk(chunk_id=_chunk_id(source, index, piece), text=piece, metadata=metadata))
    return chunks
