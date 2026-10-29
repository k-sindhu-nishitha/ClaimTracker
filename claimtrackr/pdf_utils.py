"""PDF text extraction (pypdf)."""
from __future__ import annotations

import logging
import re
from typing import BinaryIO, List, Union

from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)  # silence harmless layout warnings

PdfSource = Union[str, BinaryIO]


def _clean(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)  # layout mode pads with many spaces
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_pages(source: PdfSource, layout: bool = False) -> List[str]:
    """Return the text of each page.

    layout=True keeps line breaks and spaces - needed for small bills, where the default mode can
    glue words together ('Bodyachewithfever'). Long handbooks read better in the default mode,
    where layout mode can split words ('sha ll')."""
    reader = PdfReader(source)
    pages = []
    for page in reader.pages:
        try:
            text = (page.extract_text(extraction_mode="layout") if layout else page.extract_text()) or ""
        except TypeError:  # older pypdf without layout mode
            text = page.extract_text() or ""
        pages.append(_clean(text))
    return pages


def extract_text(source: PdfSource) -> str:
    """Text of a bill (layout mode)."""
    return "\n".join(extract_pages(source, layout=True)).strip()
