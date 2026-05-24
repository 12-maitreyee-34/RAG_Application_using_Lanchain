# EasyPaper RAG Application

## Overview

EasyPaper is a research paper ingestion and analysis platform designed for academics and researchers.
It lets users upload multiple papers and automatically extract structured content so the papers can be queried, compared, and visualized.

Goals of EasyPaper:
1. Q&A Chat: enable users to upload 3–5 research papers and ask questions across all of them together.
   Example: "What methods were used across these papers?" The RAG pipeline answers using only the uploaded papers.
2. Contradiction Detection: automatically surface where papers disagree.
   Example: "Paper A says X, Paper B says Y — contradiction found." This is surfaced automatically, not only on user request.
3. Knowledge Graph: provide a visual interactive graph on the website.
   Papers and concepts/entities are nodes, with edges showing relationships like "Paper A uses Method X", "Paper B contradicts Paper A", and "both papers reference Dataset Y." Users can explore connections across uploaded papers.


Key components:
- `RAG/ingestion.py` — PDF parsing, cleaning, abstract extraction, and section detection.
- `db/database.py` — SQLAlchemy engine and session management.
- `db/models.py` — ORM models for `Paper` and `Session`.
- `alembic/versions/0001_initial.py` — initial database migration schema.
- `test_ingestion.py` — sample script to process PDFs from the `Data/` folder.

## Database details

The application uses SQLAlchemy with a PostgreSQL backend.

Required configuration:
- `DATABASE_URL` environment variable must be set.
- The value should point to a PostgreSQL database, for example:
  `postgresql://user:password@localhost:5432/easypaper`

`db/database.py` exposes:
- `engine` — SQLAlchemy engine with connection pooling.
- `SessionLocal` — session factory for database operations.
- `Base` — declarative base for ORM models.

### `Paper` model

Defined in `db/models.py`, the `Paper` table stores:
- `doc_id` — UUID primary key.
- `filename` — PDF filename.
- `title` — extracted paper title.
- `authors` — JSONB array of author names.
- `year` — publication year.
- `doi` — DOI string.
- `abstract` — extracted abstract text.
- `sections` — JSONB list of section objects like `[{"heading": "introduction", "content": "..."}]`.
- `full_text` — cleaned full paper text.
- `page_count` — PDF page count.
- `uploaded_at` — ingestion timestamp.
- `is_chunked`, `is_embedded` — flags for later processing stages.

### `Session` model

Also in `db/models.py`, the `Session` table stores:
- `session_id` — UUID primary key.
- `paper_ids` — JSONB list of processed paper UUIDs.
- `created_at` — session creation timestamp.

## Alembic migration details

The project uses Alembic to manage schema migrations.

The initial migration is in `alembic/versions/0001_initial.py`.
It creates:
- `papers` table with PostgreSQL UUID and JSONB columns.
- `sessions` table with UUID and JSONB columns.

To run migrations:
```bash
alembic upgrade head
```

The Alembic environment is configured via `alembic.ini` and the `alembic/` directory.

## Document data structure

The ingestion pipeline builds a `Document` dataclass in `RAG/ingestion.py`.
The object contains:
- `doc_id` — internal UUID.
- `filename` — source filename.
- `title` — extracted title.
- `authors` — list of author strings.
- `year` — publication year.
- `doi` — DOI if available.
- `page_count` — number of PDF pages.
- `uploaded_at` — ingestion timestamp.
- `abstract` — abstract extracted separately.
- `full_text` — entire cleaned paper text.
- `sections` — array of section dictionaries.

Each section dictionary has:
- `heading` — normalized section label such as `abstract`, `introduction`, `methods`, etc.
- `content` — the cleaned text content for that section.

## Ingestion pipeline and difficulties

### What `RAG/ingestion.py` does

- `extract_blocks()` reads PDF text using PyMuPDF blocks instead of raw lines.
- It handles two-column layouts by splitting blocks at the page midpoint and reading left column first, then right column.
- `clean_text()` removes PDF noise such as hyphenated line breaks, page numbers, extra whitespace, and null bytes.
- `detect_sections()` identifies section headings using a list of known headings.
- `extract_metadata()` pulls title and authors from PDF metadata or from preamble text when metadata is missing.
- `extract_abstract_from_preamble()` extracts abstracts when the PDF does not explicitly label the abstract section.
- `pdf_to_document()` orchestrates the full extraction and returns the `Document` dataclass.

### Difficulties faced

#### Database and migration
- Setting up PostgreSQL and environment configuration was harder than expected because the database URL must be provided through `DATABASE_URL`.
- The initial migration had to include PostgreSQL-specific types: `UUID` and `JSONB`.
- Making sure the SQLAlchemy base, engine, and sessions were all wired correctly required careful checks of `db/database.py` and `db/models.py`.

#### Document ingestion
- Abstract and preamble were being mixed in several papers.
- Some PDFs did not label the abstract section explicitly, so the extractor had to fall back to the first long paragraph in the preamble.
- Bold headings and unusual layout caused the text extraction to merge heading and paragraph content.
- The code now handles both single-column and two-column layouts.
- The extractor also tries to avoid false positives by recognizing section headings only when they appear with punctuation or clear heading structure.

#### Section detection
- Headings like `Method`, `Methods`, or `Introduction` can appear inside normal prose, which makes pure word-based detection unreliable.
- The system must balance between catching real section headings and avoiding false splits when words appear as part of a sentence.
- PDFs with one bold paragraph and an `Introduction` heading on the next line were especially tricky.

#### PDF parsing
- PyMuPDF does not preserve semantic structure, only text blocks and font information.
- This means visual clues such as bold font can help guide heuristics but cannot be relied on directly in the current pipeline.
- Two-column PDFs required special handling so left-column text is read before right-column text.

## How to run ingestion

Put PDFs into the `Data/` folder and run:
```bash
python test_ingestion.py
```

This script processes each PDF, saves it to the database, and prints section counts.

## Notes

- The current pipeline is designed for research papers and academic PDFs.
- It may still fail on papers with unusual layout, merged headings, or very short abstracts.
- Future improvements include stronger heading heuristics, font-aware extraction, and better fallback rules for abstract detection.
