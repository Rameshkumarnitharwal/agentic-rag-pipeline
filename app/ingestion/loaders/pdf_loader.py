"""PDF loader: pypdf text layer per page, with local OCR (RapidOCR) for pages that have none."""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

from pypdf import PdfReader

from .base import Section, clean_text

logger = logging.getLogger(__name__)

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
OCR_MIN_TEXT_CHARS = 20  # pages with less extracted text than this are treated as scanned
OCR_RENDER_SCALE = 2.0  # ~144 dpi; good accuracy/speed balance for printed text

_ocr_engine = None
_ocr_lock = threading.Lock()


def _ocr_settings() -> tuple[bool, int]:
    from app.config import get_settings  # lazy: loaders are also used without app config

    settings = get_settings()
    return settings.ocr_enabled, settings.ocr_max_pages_per_document


def _engine():
    global _ocr_engine
    with _ocr_lock:
        if _ocr_engine is None:
            from rapidocr_onnxruntime import RapidOCR

            _ocr_engine = RapidOCR()
    return _ocr_engine


def ocr_page(pdf_path: Path, page_index: int) -> str:
    """Render one page and OCR it; lines are returned in reading order (top-to-bottom, left-to-right)."""
    import numpy as np
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        image = pdf[page_index].render(scale=OCR_RENDER_SCALE).to_pil().convert("RGB")
    finally:
        pdf.close()
    result, _ = _engine()(np.asarray(image))
    if not result:
        return ""
    # result: [[box(4 points), text, confidence], ...]; group into rows by the box's top y.
    lines = sorted(result, key=lambda r: (round(min(p[1] for p in r[0]) / 15), min(p[0] for p in r[0])))
    return "\n".join(text for _, text, conf in lines if float(conf) >= 0.5)


def load_pdf(path: Path) -> list[Section]:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        reader.decrypt("")  # many "encrypted" PDFs only have an empty owner password

    ocr_enabled, ocr_budget = _ocr_settings()
    sections: list[Section] = []
    ocr_pages = 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # malformed page streams shouldn't kill the whole document
            logger.warning("Text extraction failed on page %d of %s: %s", page_number, path.name, exc)
            raw = ""
        metadata = {"page": page_number}
        if len(raw.strip()) < OCR_MIN_TEXT_CHARS and ocr_enabled and ocr_pages < ocr_budget:
            try:
                raw = ocr_page(path, page_number - 1)
                ocr_pages += 1
                metadata["ocr"] = True
            except Exception as exc:
                logger.warning("OCR failed on page %d of %s: %s", page_number, path.name, exc)
        text = clean_text(_HYPHEN_BREAK.sub(r"\1\2", raw))
        if text:
            sections.append(Section(text=text, metadata=metadata))

    if ocr_pages:
        logger.info("OCR'd %d page(s) of %s", ocr_pages, path.name)
        if ocr_pages >= ocr_budget:
            logger.warning("%s hit the OCR page limit (%d); remaining scanned pages skipped", path.name, ocr_budget)
    if not sections:
        logger.warning("No text found in %s (scanned PDF with OCR disabled?)", path.name)
    return sections
