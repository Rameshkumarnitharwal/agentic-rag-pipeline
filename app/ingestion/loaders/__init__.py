"""Local document parsers — no external OCR or parsing services."""

from __future__ import annotations

from pathlib import Path

from .base import Loader, Section
from .docx_loader import load_docx
from .html_loader import load_html
from .pdf_loader import load_pdf
from .pptx_loader import load_pptx
from .txt_loader import load_txt

LOADERS: dict[str, Loader] = {
    ".pdf": load_pdf,
    ".html": load_html,
    ".htm": load_html,
    ".txt": load_txt,
    ".md": load_txt,
    ".docx": load_docx,
    ".pptx": load_pptx,
}

SUPPORTED_EXTENSIONS = frozenset(LOADERS)


class UnsupportedFileType(ValueError):
    pass


def load_document(path: str | Path) -> list[Section]:
    """Parse a file into Sections, choosing the loader by file extension."""
    path = Path(path)
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise UnsupportedFileType(f"Unsupported file type: {path.suffix or path.name}")
    return loader(path)


__all__ = ["LOADERS", "SUPPORTED_EXTENSIONS", "Section", "UnsupportedFileType", "load_document"]
