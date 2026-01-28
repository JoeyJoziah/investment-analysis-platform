"""merge migration heads

Revision ID: a20ad12e7a8d
Revises: 010, c849a2ab3b24
Create Date: 2026-01-27 22:02:17.556970

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a20ad12e7a8d'
down_revision = ('010', 'c849a2ab3b24')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass