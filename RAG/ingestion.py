# extract_blocks()    →  reads PDF paragraph by paragraph (not line by line)
#                        so sentences never break mid-way
#                        FIX: handles two-column layouts (IEEE papers etc)

# clean_text()        →  removes noise: page numbers, null bytes,
#                        hyphenated line breaks, extra spaces

# detect_sections()   →  finds headings like "introduction", "methods" etc
#                        skips references and acknowledgements
#                        falls back to one "body" section if nothing found

# extract_abstract_from_preamble() → extracts abstract when no heading exists
#                        FIX: strips IEEE copyright lines before evaluating
# remove_references_from_content() → strips reference lists mixed into sections
#                        FIX: now also handles IEEE-style [1] [2] references
# extract_authors_from_text()      → NEW: extracts authors from preamble text

# extract_metadata()  →  gets title, authors, year from PyMuPDF
#                        FIX: now accepts sections to extract authors from preamble
#                        falls back to scanning the text if metadata is empty

# pdf_to_document()   →  orchestrates all the above into one clean Document


import fitz
import re
import uuid
import os
from datetime import datetime
from dataclasses import dataclass
from db.database import SessionLocal
from db.models import Paper, Session

# ── 1. Document Dataclass ─────────────────────────────────
@dataclass
class Document:
    # Identity
    doc_id:      str

    # Metadata
    filename:    str
    title:       str
    authors:     list[str]
    year:        int
    doi:         str
    page_count:  int
    uploaded_at: datetime

    # Content
    abstract:    str        # extracted separately — critical for contradiction detection
    full_text:   str        # entire cleaned text — fallback for chunking
    sections:    list[dict] # [{"heading": "introduction", "content": "..."}]


# ── 2. Known section headings ─────────────────────────────
# FIX: added "literature survey" and "experimental results" for IEEE-style papers
SECTION_HEADINGS = [
    "abstract", "introduction", "background", "related work",
    "literature review", "literature survey",
    "materials and methods", "methodology",
    "methods", "approach", "proposed method", "proposed work", "experiments",
    "experimental setup", "experimental results",
    "results", "results and discussion",
    "discussion", "conclusion", "conclusions",
    "future work", "limitations"
]

SKIP_SECTIONS = {
    "references", "bibliography",
    "acknowledgements", "acknowledgments"
}


# ── 3. Heading detector ───────────────────────────────────
def is_section_heading(line: str) -> str | None:
    """
    Returns normalized heading name if line is a heading, else None.
    Handles: ALL CAPS, numbered (1. / 1.1 / I.), mixed case, short lines.

    FIX: added word count check — real headings have max 6 words.
    This prevents mid-paragraph words like "methods" being detected as headings.
    """
    stripped = line.strip()

    # headings are short — skip long lines
    if not stripped or len(stripped) > 80:
        return None

    # FIX: if more than 6 words, it's a sentence not a heading
    if len(stripped.split()) > 6:
        return None

    # remove leading numbering: "1.", "2.1", "I.", "A."
    cleaned = re.sub(r'^(\d+\.)*\d+\.?\s*', '', stripped)
    cleaned = re.sub(r'^[IVXLC]+\.\s*', '', cleaned)
    cleaned = cleaned.strip().lower()

    for h in SECTION_HEADINGS:
        if re.match(rf'^{re.escape(h)}(?:\s|[:\-\–\—\.]|$)', cleaned):
            return h

    return None


# ── 4. Extract text using blocks ──────────────────────────
def extract_blocks(pdf_path: str) -> str:
    """
    Uses PyMuPDF blocks instead of raw lines.
    Blocks group text at paragraph level — sentences stay intact.

    FIX: handles two-column layouts (IEEE conference papers etc).
    Detects page midpoint and reads left column fully before right column.
    Single-column papers work exactly as before.
    """
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        blocks = page.get_text("blocks")
        if not blocks:
            continue

        page_width = page.rect.width
        mid_x = page_width / 2

        # separate into left and right column blocks
        # 20px tolerance around midpoint to avoid splitting wide blocks
        left_blocks  = [b for b in blocks if b[0] < mid_x - 20]
        right_blocks = [b for b in blocks if b[0] >= mid_x - 20]

        # sort each column top→bottom independently
        left_blocks  = sorted(left_blocks,  key=lambda b: b[1])
        right_blocks = sorted(right_blocks, key=lambda b: b[1])

        # read left column fully first, then right column
        # for single-column papers, most blocks will be in left_blocks
        # and right_blocks will be near-empty — works correctly either way
        for block in left_blocks + right_blocks:
            text = block[4].strip()
            if text:
                full_text += text + "\n\n"

    return full_text


# ── 5. Clean text ─────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r'-\n', '', text)          # fix hyphenated line breaks: "computa-\ntion" → "computation"
    text = re.sub(r'\b(Page\s*\d+(\s*of\s*\d+)?)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)  # collapse excess blank lines
    text = re.sub(r'[ \t]+', ' ', text)     # collapse spaces/tabs
    text = text.replace('\x00', '')          # remove null bytes
    return text.strip()


# ── 6. Remove references sneaking into other sections ─────
def remove_references_from_content(content: str) -> str:
    """
    Safety net for when the references heading isn't detected.
    If a section's content starts looking like a bibliography,
    cut everything from that point onwards.

    Reference lines look like:
    - contain 'doi:'
    - contain (2021) year-in-brackets pattern
    - contain 'Author, F.' name patterns
    - FIX: IEEE style [1], [2], [3] numbered references
    """
    lines = content.split('\n')
    consecutive_ref_lines = 0
    ref_start = None

    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        if not line_lower:
            continue

        looks_like_ref = (
            'doi:' in line_lower or
            re.search(r'\(\d{4}\)', line) or               # (2021) pattern
            re.search(r'[A-Z][a-z]+,\s[A-Z]\.', line) or  # "Deng, R." pattern
            line_lower.startswith('frontiers in') or        # journal footer lines
            re.match(r'^\[\d+\]', line.strip())             # FIX: [1] [2] IEEE style
        )

        if looks_like_ref:
            consecutive_ref_lines += 1
        else:
            consecutive_ref_lines = 0

        # 3 consecutive reference-looking lines = we've hit the references
        if consecutive_ref_lines >= 3:
            ref_start = i - 2
            break

    if ref_start is not None and ref_start > 0:
        return '\n'.join(lines[:ref_start]).strip()

    return content


# ── 7. Extract abstract from preamble ────────────────────
def extract_abstract_from_preamble(preamble_content: str) -> str:
    """
    Many journal papers don't label the abstract with a heading.
    It's the first long dense prose paragraph before Introduction.

    FIX: strips IEEE copyright/footer lines before evaluating paragraphs.
    FIX: raised digit threshold from 5 to 7 to allow citation numbers like [1].
    """
    paragraphs = [p.strip() for p in preamble_content.split('\n\n') if p.strip()]

    for para in paragraphs:
        # FIX: strip IEEE copyright and access lines before evaluating
        clean_para = re.sub(r'\d{3}-\d-\d+-\d+-\d+/\d+/\$[\d.]+.*?IEEE', '', para).strip()
        clean_para = re.sub(r'©\d{4} IEEE', '', clean_para).strip()
        clean_para = re.sub(r'Authorized licensed use.*?apply\.', '', clean_para, flags=re.DOTALL).strip()

        words = clean_para.split()

        # abstract is typically 30-500 words
        if len(words) < 30 or len(words) > 500:
            continue

        # skip metadata patterns
        if '@' in clean_para:                                    continue  # email
        if clean_para.lower().startswith('http'):                continue  # URL
        if 'doi:' in clean_para.lower():                        continue  # DOI line
        if re.search(r'\d{7,}', clean_para):                    continue  # FIX: raised to 7 digits
        if 'frontiers' in clean_para.lower()[:50]:              continue  # journal name
        if re.match(r'^[A-Z][a-z]+ [A-Z]', clean_para[:30]):  continue  # "Ruoling Deng" author pattern

        # looks like real prose — return it
        return clean_para

    return ""


# ── 8. Extract authors from preamble text ────────────────
def extract_authors_from_text(preamble_content: str) -> list[str]:
    """
    NEW: Extracts author names from preamble text.

    Tries multiple patterns:
    1. "Author1, Author2, Author3 and Author4" — comma+and separated (Frontiers style)
    2. "Author1    Author2" side by side — IEEE two-column style
    3. Returns [] if nothing found — metadata stays empty

    Author lines appear after the title, before affiliations/abstract.
    They are short lines with proper names (capital letters).
    """
    lines = [l.strip() for l in preamble_content.split('\n') if l.strip()]

    for line in lines:

        # skip very long lines — not author lines
        if len(line) > 150:
            continue

        # skip lines that look like affiliations, emails, or metadata
        if '@' in line:                          continue
        if 'university' in line.lower():         continue
        if 'department' in line.lower():         continue
        if 'institute' in line.lower():          continue
        if 'college' in line.lower():            continue
        if 'laboratory' in line.lower():         continue
        if 'email' in line.lower():              continue
        if 'doi' in line.lower():                continue
        if line.lower().startswith('http'):      continue
        if re.search(r'\d{4}', line):            continue  # lines with years/numbers

        # Pattern 1 — "Name1, Name2, Name3 and Name4" (Frontiers/journal style)
        if re.search(r'[A-Z][a-z]+.*(?:,|and)\s+[A-Z][a-z]+', line):
            parts = re.split(r',|\band\b', line)
            authors = []
            for p in parts:
                # remove affiliation numbers like "1", "2*", "*"
                name = re.sub(r'\s*\d+\*?\s*', ' ', p).strip()
                name = re.sub(r'\*', '', name).strip()
                if name and len(name.split()) >= 2:
                    authors.append(name)
            if len(authors) >= 1:
                return authors

        # Pattern 2 — "S.Ramesh    D.vydeki" IEEE two-column side-by-side style
        if re.search(r'[A-Z][a-z.]+\s{2,}[A-Z][a-z.]+', line):
            parts = re.split(r'\s{2,}', line)
            authors = [p.strip() for p in parts if p.strip() and re.match(r'[A-Z]', p.strip())]
            if len(authors) >= 2:
                return authors

    return []


# ── 9. Detect sections ────────────────────────────────────
def detect_sections(text: str) -> list[dict]:
    """
    Splits cleaned full_text into sections based on heading detection.
    Skips references, acknowledgements.
    Strips reference lists that sneak into other sections.
    Falls back to one section 'body' if no headings found.
    """
    lines = text.split('\n')
    sections = []
    current_heading = "preamble"
    current_content = []
    skip_current = False

    for line in lines:
        heading = is_section_heading(line)

        if heading:
            # save previous section unless skipped
            if current_content and not skip_current:
                content = " ".join(current_content).strip()
                # FIX: strip any references that snuck into this section
                content = remove_references_from_content(content)
                if content:
                    sections.append({
                        "heading": current_heading,
                        "content": content
                    })

            current_heading = heading
            current_content = []
            skip_current = heading in SKIP_SECTIONS

        else:
            if not skip_current:
                current_content.append(line.strip())

    # save last section
    if current_content and not skip_current:
        content = " ".join(current_content).strip()
        # FIX: strip references from last section too (conclusion often has them)
        content = remove_references_from_content(content)
        if content:
            sections.append({
                "heading": current_heading,
                "content": content
            })

    # fallback — if nothing detected, store everything as 'body'
    if not sections:
        sections = [{"heading": "body", "content": text.strip()}]

    return sections


# ── 10. Extract metadata ──────────────────────────────────
def extract_metadata(fitz_doc: fitz.Document, full_text: str, sections: list[dict]) -> dict:
    """
    FIX: now accepts sections parameter so authors can be
    extracted from preamble text when PyMuPDF metadata is empty.
    """
    meta = fitz_doc.metadata

    # Title — try PyMuPDF first, else first long line of text
    title = meta.get("title", "").strip()
    if not title:
        for line in full_text.split('\n'):
            line = line.strip()
            if len(line) > 20 and not line.lower().startswith("http"):
                title = line
                break
    title = title or "Unknown Title"

    # Authors — try PyMuPDF metadata first
    authors_raw = meta.get("author", "").strip()
    authors = [a.strip() for a in authors_raw.split(",") if a.strip()]

    # FIX: if metadata empty, extract from preamble text
    if not authors:
        preamble_content = ""
        for s in sections:
            if s["heading"] == "preamble":
                preamble_content = s["content"]
                break
        if preamble_content:
            authors = extract_authors_from_text(preamble_content)

    # Year — from metadata creation date, else scan first 500 chars of text
    year = None
    creation = meta.get("creationDate", "")
    if creation and len(creation) >= 6:
        try:
            year = int(creation[2:6])
        except ValueError:
            pass
    if not year:
        match = re.search(r'\b(19|20)\d{2}\b', full_text[:500])
        if match:
            year = int(match.group())

    return {
        "title":   title,
        "authors": authors,
        "year":    year,
        "doi":     ""   # PDFs rarely store DOI in metadata
    }


# ── 11. Main function: PDF → Document ────────────────────
def pdf_to_document(pdf_path: str) -> Document:
    fitz_doc = fitz.open(pdf_path)

    # Extract → Clean → Structure
    raw_text  = extract_blocks(pdf_path)
    clean     = clean_text(raw_text)
    sections  = detect_sections(clean)

    # FIX: pass sections to extract_metadata so authors can be pulled from preamble
    meta      = extract_metadata(fitz_doc, clean, sections)

    # Pull abstract — try sections first (papers that label it)
    abstract = ""
    for s in sections:
        if s["heading"] == "abstract":
            abstract = s["content"]
            break

    # FIX: fallback — extract from preamble content (unlabelled abstracts)
    if not abstract:
        for s in sections:
            if s["heading"] == "preamble":
                abstract = extract_abstract_from_preamble(s["content"])
                break

    # FIX: last fallback — first 300 words of introduction (better than full_text)
    if not abstract:
        for s in sections:
            if s["heading"] == "introduction":
                abstract = " ".join(s["content"].split()[:300])
                break

    return Document(
        doc_id      = str(uuid.uuid4()),
        filename    = os.path.basename(pdf_path),
        title       = meta["title"],
        authors     = meta["authors"],
        year        = meta["year"],
        doi         = meta["doi"],
        page_count  = fitz_doc.page_count,
        uploaded_at = datetime.utcnow(),
        abstract    = abstract,
        full_text   = clean,
        sections    = sections,
    )


# ── 12. Save to DB ────────────────────────────────────────
def save_document(document: Document):
    db = SessionLocal()
    try:
        paper = Paper(
            doc_id      = document.doc_id,
            filename    = document.filename,
            title       = document.title,
            authors     = document.authors,
            year        = document.year,
            doi         = document.doi,
            page_count  = document.page_count,
            uploaded_at = document.uploaded_at,
            abstract    = document.abstract,
            full_text   = document.full_text,
            sections    = document.sections,
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