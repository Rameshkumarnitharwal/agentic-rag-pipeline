"""HTML loader using BeautifulSoup — strips page chrome, keeps headings, code and tables."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from .base import Section, clean_text, rows_to_markdown
from .txt_loader import read_text

_DROP_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form", "iframe", "svg", "button"]
_BLOCK_TAGS = {"p", "li", "dt", "dd", "blockquote", "figcaption", "caption"}
_HEADINGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}


def _table_to_markdown(table: Tag) -> str:
    rows = [
        [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
        for tr in table.find_all("tr")
    ]
    return rows_to_markdown(rows)


def _blocks(node: Tag) -> list[str]:
    """Walk the DOM and emit one text block per heading/paragraph/code/table."""
    blocks: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            text = " ".join(str(child).split())
            if text:
                blocks.append(text)
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name in _HEADINGS:
            blocks.append(f"{_HEADINGS[name]} {child.get_text(' ', strip=True)}")
        elif name == "pre":
            blocks.append("```\n" + child.get_text().strip("\n") + "\n```")
        elif name == "table":
            blocks.append(_table_to_markdown(child))
        elif name in _BLOCK_TAGS:
            text = " ".join(child.get_text(" ", strip=True).split())
            blocks.append(f"- {text}" if name == "li" else text)
        else:
            blocks.extend(_blocks(child))
    return [b for b in blocks if b.strip()]


def load_html(path: Path) -> list[Section]:
    soup = BeautifulSoup(read_text(path), "lxml")
    title = soup.title.get_text(strip=True) if soup.title else None
    for tag in soup(_DROP_TAGS):
        tag.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    text = clean_text("\n\n".join(_blocks(root)))
    if not text:
        return []
    return [Section(text=text, metadata={"title": title} if title else {})]
