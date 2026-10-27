"""DOCX loader using python-docx — paragraphs and tables in document order."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from .base import Section, clean_text, rows_to_markdown

# Paragraph styles / fonts Word uses for code. Exported web pages typically use "HTML Preformatted",
# with one paragraph per code line.
_CODE_STYLES = ("preformatted", "code", "source", "macro text")
_MONO_FONTS = ("consolas", "courier", "menlo", "monaco", "lucida console", "source code", "cascadia")


def _is_code(paragraph: Paragraph) -> bool:
    style = (paragraph.style.name or "").lower() if paragraph.style is not None else ""
    if any(marker in style for marker in _CODE_STYLES):
        return True
    fonts = {(run.font.name or "").lower() for run in paragraph.runs if run.text.strip()}
    return bool(fonts) and all(any(mono in font for mono in _MONO_FONTS) for font in fonts)


def _paragraph_block(paragraph: Paragraph) -> str:
    text = paragraph.text.strip()
    if not text:
        return ""
    style = (paragraph.style.name or "").lower() if paragraph.style is not None else ""
    if style == "title":
        return f"# {text}"
    if style.startswith("heading"):
        level = style.removeprefix("heading").strip()
        return f"{'#' * (int(level) + 1 if level.isdigit() else 2)} {text}"
    if "list" in style:
        return f"- {text}"
    return text


def _table_block(table: Table) -> str:
    rows: list[list[str]] = []
    for row in table.rows:
        cells: list[str] = []
        previous = None
        for cell in row.cells:
            # Merged cells repeat the same cell object; emit it once.
            if previous is not None and cell._tc is previous:
                continue
            previous = cell._tc
            cells.append(cell.text)
        rows.append(cells)
    return rows_to_markdown(rows)


def load_docx(path: Path) -> list[Section]:
    document = Document(str(path))
    blocks: list[str] = []
    code_lines: list[str] = []

    def flush_code() -> None:
        # Consecutive code paragraphs become one fenced block (indentation kept, blank lines dropped
        # so the block stays a single paragraph for the chunker).
        if any(line.strip() for line in code_lines):
            body = "\n".join(line.rstrip() for line in code_lines if line.strip())
            blocks.append(f"```\n{body}\n```")
        code_lines.clear()

    for item in document.iter_inner_content():
        if isinstance(item, Paragraph) and _is_code(item):
            code_lines.append(item.text)
            continue
        flush_code()
        block = _table_block(item) if isinstance(item, Table) else _paragraph_block(item)
        if block:
            blocks.append(block)
    flush_code()

    text = clean_text("\n\n".join(blocks))
    title = document.core_properties.title or None
    if not text:
        return []
    return [Section(text=text, metadata={"title": title} if title else {})]
