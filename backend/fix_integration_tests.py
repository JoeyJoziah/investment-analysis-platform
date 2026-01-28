#!/usr/bin/env python3
"""Fix integration tests to use unified_models instead of tables"""

import re
from pathlib import Path

# Files to fix
files_to_fix = [
    "tests/integration/test_stock_to_analysis_flow.py",
    "tests/integration/test_gdpr_data_lifecycle.py",
    "tests/integration/test_phase3_integration.py",
]

# Common fixture template
exchange_fixture = '''
@pytest_asyncio.fixture
async def nasdaq_exchange(db_session: AsyncSession):
    """Create NASDAQ exchange for testing."""
    exchange = Exchange(
        code="NASDAQ",
        name="NASDAQ Stock Market",
        country="US",
        currency="USD",
        timezone="America/New_York"
    )
    db_session.add(exchange)
    await db_session.commit()
    await db_session.refresh(exchange)
    return exchange


@pytest_asyncio.fixture
async def technology_sector(db_session: AsyncSession):
    """Create Technology sector for testing."""
    sector = Sector(
        name="Technology",
        description="Technology sector"
    )
    db_session.add(sector)
    await db_session.commit()
    await db_session.refresh(sector)
    return sector
'''

for filepath in files_to_fix:
    print(f"Processing {filepath}...")

    with open(filepath, 'r') as f:
        content = f.read()

    # Replace imports
    content = re.sub(
        r'from backend\.models\.tables import',
        'from backend.models.unified_models import',
        content
    )

    # Add Exchange and Sector to imports if not present
    if 'Exchange' not in content:
        content = re.sub(
            r'(from backend\.models\.unified_models import[^)]+)',
            r'\1, Exchange, Sector',
            content
        )

    # Replace Stock fixtures
    # Pattern 1: exchange="NASDAQ"
    content = re.sub(
        r'exchange="[^"]+",',
        'exchange_id=nasdaq_exchange.id,',
        content
    )

    # Pattern 2: sector="Technology"
    content = re.sub(
        r'sector="[^"]+",',
        'sector_id=technology_sector.id,',
        content
    )

    # Pattern 3: asset_type=AssetTypeEnum.STOCK
    content = re.sub(
        r'asset_type=AssetTypeEnum\.STOCK',
        'asset_type="stock"',
        content
    )

    # Find first @pytest_asyncio.fixture and inject exchange/sector fixtures before it
    if 'nasdaq_exchange' not in content:
        first_fixture_match = re.search(r'(@pytest_asyncio\.fixture)', content)
        if first_fixture_match:
            pos = first_fixture_match.start()
            content = content[:pos] + exchange_fixture + '\n' + content[pos:]

    # Update fixture signatures to include nasdaq_exchange and technology_sector
    # Pattern: async def sample_stock(db_session: AsyncSession):
    content = re.sub(
        r'async def ([a-z_]+stock[a-z_]*)\(db_session: AsyncSession\):',
        r'async def \1(db_session: AsyncSession, nasdaq_exchange: Exchange, technology_sector: Sector):',
        content
    )

    with open(filepath, 'w') as f:
        f.write(content)

    print(f"  ✓ Fixed {filepath}")

print("\n✅ All integration tests fixed!")
