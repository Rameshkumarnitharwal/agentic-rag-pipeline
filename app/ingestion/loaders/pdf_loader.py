"""PDF loader using pypdf — one Section per page, text layer only (no OCR)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pypdf import PdfReader

from .base import Section, clean_text

logger = logging.getLogger(__name__)

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")


def load_pdf(path: Path) -> list[Section]:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        reader.decrypt("")  # many "encrypted" PDFs only have an empty owner password

    sections: list[Section] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # malformed page streams shouldn't kill the whole document
            logger.warning("Skipping page %d of %s: %s", page_number, path.name, exc)
            continue
        text = clean_text(_HYPHEN_BREAK.sub(r"\1\2", raw))
        if text:
            sections.append(Section(text=text, metadata={"page": page_number}))

    if not sections:
        logger.warning("No text layer found in %s (scanned PDF?)", path.name)
    return sections
