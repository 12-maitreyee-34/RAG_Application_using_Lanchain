"""initial migration

Revision ID: 000000000001
Revises: 
Create Date: 2026-05-22 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '000000000001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'papers',
        sa.Column('doc_id', postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column('filename', sa.String(), nullable=True),
        sa.Column('title', sa.String(), nullable=True),
        sa.Column('authors', postgresql.JSONB(), nullable=True),
        sa.Column('year', sa.Integer(), nullable=True),
        sa.Column('doi', sa.String(), nullable=True),
        sa.Column('abstract', sa.Text(), nullable=True),
        sa.Column('sections', postgresql.JSONB(), nullable=True),
        sa.Column('full_text', sa.Text(), nullable=True),
        sa.Column('page_count', sa.Integer(), nullable=True),
        sa.Column('uploaded_at', sa.DateTime(), nullable=True),
        sa.Column('is_chunked', sa.Boolean(), nullable=True),
        sa.Column('is_embedded', sa.Boolean(), nullable=True),
    )

    op.create_table(
        'sessions',
        sa.Column('session_id', postgresql.UUID(as_uuid=False), primary_key=True, nullable=False),
        sa.Column('paper_ids', postgresql.JSONB(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('sessions')
    op.drop_table('papers')
