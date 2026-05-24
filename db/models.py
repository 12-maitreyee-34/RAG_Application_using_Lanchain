from datetime import datetime
import uuid

from sqlalchemy import Column, String, Integer, Text, Boolean, DateTime
from sqlalchemy.dialects.postgresql import JSONB, UUID

from db.database import Base

class Paper(Base):
    __tablename__ = "papers"

    doc_id      = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename    = Column(String)
    title       = Column(String)
    authors     = Column(JSONB)       # ["Author A", "Author B"]
    year        = Column(Integer)
    doi         = Column(String)
    abstract    = Column(Text)
    sections    = Column(JSONB)       # [{"heading": "Methods", "content": "..."}]
    full_text   = Column(Text)
    page_count  = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    is_chunked  = Column(Boolean, default=False)
    is_embedded = Column(Boolean, default=False)

class Session(Base):
    __tablename__ = "sessions"

    session_id  = Column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4()))
    paper_ids   = Column(JSONB)       # ["uuid1", "uuid2", "uuid3"]
    created_at  = Column(DateTime, default=datetime.utcnow)