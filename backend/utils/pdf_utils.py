import re
import fitz  # PyMuPDF


def is_scanned_pdf(doc: fitz.Document, min_char_threshold: int = 100) -> bool:
    """
    Detect if a PDF is image-only (scanned) by checking total extracted character count.
    Returns True if the PDF appears to be scanned.
    """
    total_chars = sum(len(page.get_text("text").strip()) for page in doc)
    return total_chars < min_char_threshold


def get_page_count(doc: fitz.Document) -> int:
    return len(doc)


def extract_section_header(text: str) -> str | None:
    """
    Use regex heuristics to detect a section heading at the start of a chunk.
    Patterns covered:
      - Numbered sections: "1. Introduction", "2.3 Methods"
      - ALL CAPS short lines (≤ 60 chars)
      - Title Case short lines with no trailing punctuation
    """
    lines = text.strip().split("\n")
    for line in lines[:3]:  # check only the first 3 lines
        line = line.strip()
        if not line or len(line) > 80:
            continue
        # Numbered heading: 1. Foo, 1.1 Foo, Chapter 1
        if re.match(r"^(\d+[\.\d]*\s+|Chapter\s+\d+\s*:?\s*)[A-Z]", line):
            return line
        # ALL CAPS heading
        if line.isupper() and 3 < len(line) <= 60:
            return line
        # Title Case line (no period at end, short)
        words = line.split()
        if (
            len(words) >= 2
            and len(line) <= 60
            and not line.endswith(".")
            and sum(1 for w in words if w[0].isupper()) >= len(words) * 0.6
        ):
            return line
    return None


def detect_chunk_type(text: str) -> str:
    """
    Classify a chunk as 'list', 'table', or 'paragraph'.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return "paragraph"

    # Table: majority of lines contain pipe character
    pipe_lines = sum(1 for l in lines if "|" in l)
    if pipe_lines / len(lines) > 0.4:
        return "table"

    # List: majority of lines start with bullet or number markers
    list_lines = sum(
        1 for l in lines if re.match(r"^[-•*\u2022\u2013\u2014]|\d+[.)]\s", l)
    )
    if list_lines / len(lines) > 0.3:
        return "list"

    return "paragraph"
