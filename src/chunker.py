from dataclasses import dataclass

from src.config import settings
from src.pdf_processor import PageContent


@dataclass
class Chunk:
    text: str
    source_file: str
    page_number: int
    chunk_index: int 


def chunk_pages(pages: list[PageContent]) -> list[Chunk]:
    """
    Split page text into overlapping fixed-size character chunks.
    Each chunk retains its source file and page number for citation.
    """
    chunks = []
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    step = size - overlap

    for page in pages:
        text = page.text
        chunk_index = 0
        start = 0

        while start < len(text):
            chunk_text = text[start: start + size].strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                ))
            start += step
            chunk_index += 1

    return chunks
