"""create document_chunks table

Revision ID: 0001
Revises:
Create Date: 2026-08-01 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            embedding vector(1536)
        );
        """
    )
    op.create_index(
        "ix_document_chunks_content",
        "document_chunks",
        ["content"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_content", table_name="document_chunks")
    op.execute("DROP TABLE IF EXISTS document_chunks;")
