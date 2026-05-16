import fitz                    # PyMuPDF
import re
import uuid
import os
from dataclasses import dataclass, field
from db.database import SessionLocal
from db.models import Paper, Session

# ── 1. Dataclass ──────────────────────────────────────────
@dataclass
class Document:
    doc_id:    str
    filename:  str
    title:     str
    authors:   list[str]
    year:      int
    doi:       str
    abstract:  str
    sections:  list[dict]   # [{"heading": "...", "content": "..."}]
    full_text: str
    page_count: int

# ── 2. Cleaning ───────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r'\b(Page\s*\d+(\s*of\s*\d+)?)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'-\n', '', text)          # fix hyphenated line breaks
    text = re.sub(r'\n{3,}', '\n\n', text)  # collapse excess blank lines
    text = re.sub(r'[ \t]+', ' ', text)     # collapse spaces
    text = text.replace('\x00', '')          # null bytes
    return text.strip()

# ── 3. Section detection ──────────────────────────────────
SECTION_HEADINGS = [
    "abstract", "introduction", "background", "related work",
    "methodology", "methods", "approach", "experiments",
    "results", "discussion", "conclusion", "references"
]

def detect_sections(text: str) -> list[dict]:
    lines = text.split('\n')
    sections = []
    current_heading = "preamble"
    current_content = []

    for line in lines:
        stripped = line.strip().lower()
        # check if this line IS a section heading
        if any(stripped == h or stripped.startswith(h + ' ') for h in SECTION_HEADINGS):
            # save the previous section
            if current_content:
                sections.append({
                    "heading": current_heading,
                    "content": " ".join(current_content).strip()
                })
            current_heading = line.strip()
            current_content = []
        else:
            current_content.append(line.strip())

    # save last section
    if current_content:
        sections.append({
            "heading": current_heading,
            "content": " ".join(current_content).strip()
        })

    return sections

# ── 4. Metadata extraction ────────────────────────────────
def extract_metadata(doc: fitz.Document, sections: list[dict]) -> dict:
    # PyMuPDF gives us basic metadata
    meta = doc.metadata

    # abstract — look for it in sections first
    abstract = ""
    for s in sections:
        if "abstract" in s["heading"].lower():
            abstract = s["content"]
            break

    return {
        "title":    meta.get("title", "Unknown Title"),
        "authors":  [a.strip() for a in meta.get("author", "").split(",") if a.strip()],
        "year":     int(meta.get("creationDate", "D:20000101")[ 2:6]) if meta.get("creationDate") else None,
        "doi":      "",   # PDFs rarely have DOI in metadata — you can add manual input later
        "abstract": abstract,
    }

# ── 5. Main function: PDF → Document ─────────────────────
def pdf_to_document(pdf_path: str) -> Document:
    doc = fitz.open(pdf_path)

    # Extract all text
    raw_text = ""
    for page in doc:
        raw_text += page.get_text()

    # Clean
    clean = clean_text(raw_text)

    # Detect sections
    sections = detect_sections(clean)

    # Metadata
    meta = extract_metadata(doc, sections)

    return Document(
        doc_id     = str(uuid.uuid4()),
        filename   = os.path.basename(pdf_path),
        title      = meta["title"],
        authors    = meta["authors"],
        year       = meta["year"],
        doi        = meta["doi"],
        abstract   = meta["abstract"],
        sections   = sections,
        full_text  = clean,
        page_count = doc.page_count,
    )

# ── 6. Save to DB ─────────────────────────────────────────
def save_document(document: Document):
    db = SessionLocal()
    try:
        paper = Paper(
            doc_id     = document.doc_id,
            filename   = document.filename,
            title      = document.title,
            authors    = document.authors,
            year       = document.year,
            doi        = document.doi,
            abstract   = document.abstract,
            sections   = document.sections,
            full_text  = document.full_text,
            page_count = document.page_count,
        )
        db.add(paper)
        db.commit()
    finally:
        db.close()

def save_session(doc_ids: list[str]) -> str:
    db = SessionLocal()
    try:
        session = Session(paper_ids=doc_ids)
        db.add(session)
        db.commit()
        return str(session.session_id)
    finally:
        db.close()