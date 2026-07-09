"""Wave 3: agents technical fail-loud + auth register username contract (#208)."""
from __future__ import annotations

import inspect
import os

os.environ.setdefault("SECRET_KEY", "test-secret-wave3")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-wave3")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("TESTING", "True")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("MASTER_SECRET_KEY", "test-master-wave3")


def test_register_source_sets_username_from_email():
    """#208 item 3: register must set username so sub/username lookup works."""
    from backend.api.routers import auth as auth_mod

    source = inspect.getsource(auth_mod.register)
    assert "username=user.email" in source
    assert "create_tokens" in inspect.getsource(auth_mod) or "_issue_access_token" in source


def test_get_loading_stats_uses_allowlist_and_sqlalchemy_core():
    """#208 item 2 / #242-C: no caller-controlled f-string table names."""
    import backend.etl.data_loader as loader_mod

    source = inspect.getsource(loader_mod.DataLoader.get_loading_stats)
    assert "_STATS_TABLE_ALLOWLIST" in source
    assert "select(func.count())" in source or "func.count" in source
    assert "select_from(sa_table" in source or "sa_table(table)" in source
    # Must not interpolate arbitrary table into text() for the allowlist loop
    assert 'text(f"SELECT COUNT(*) FROM {table}' not in source
    assert "text(f'SELECT COUNT(*) FROM {table}" not in source


def test_agents_technical_no_inline_np_random_seed_on_hot_path():
    """Production path must load real OHLCV; synthetic only under DEMO_MODE helper."""
    import backend.services.agents_service as agents

    tech_src = inspect.getsource(agents.run_technical_analysis)
    assert "_load_ohlcv_frame" in tech_src
    assert "ModelUnavailableError" in tech_src
    assert "np.random.seed" not in tech_src
    synth_src = inspect.getsource(agents._synthetic_ohlcv_demo_only)
    assert "np.random" in synth_src
