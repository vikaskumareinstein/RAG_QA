from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass
class PageContent:
    text: str
    page_number: int
    source_file: str


def extract_pages(pdf_path: str, display_name: str | None = None) -> list[PageContent]:
    """Extract text from each page of a PDF, skipping blank pages."""
    filename = display_name or Path(pdf_path).name
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(PageContent(
                    text=text.strip(),
                    page_number=i + 1,
                    source_file=filename,
                ))

    return pages
