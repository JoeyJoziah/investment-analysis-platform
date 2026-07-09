"""Wave 12: oversized split, S3 backups, canary, Terraform, SSL, ML, staging contracts."""
from __future__ import annotations

import inspect
import os
from pathlib import Path

os.environ.setdefault("SECRET_KEY", "test-secret-wave12")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-wave12")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("TESTING", "True")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("MASTER_SECRET_KEY", "test-master-wave12")


def test_security_config_under_800_lines_with_validators_extracted():
    """#99: security_config split; validators module extracted."""
    cfg = Path("backend/security/security_config.py")
    validators = Path("backend/security/security_validators.py")
    assert cfg.exists() and validators.exists()
    cfg_lines = len(cfg.read_text(encoding="utf-8").splitlines())
    assert cfg_lines <= 800, f"security_config still oversized: {cfg_lines}"
    assert "PasswordValidator" in validators.read_text(encoding="utf-8")
    assert "SecurityScanner" in validators.read_text(encoding="utf-8")

    from backend.security.security_config import (
        PasswordValidator,
        SecurityConfig,
        SecurityScanner,
    )

    assert hasattr(SecurityConfig, "PASSWORD_MIN_LENGTH")
    result = PasswordValidator.validate_password("Aa1!aaaa")
    assert "valid" in result
    assert callable(SecurityScanner.scan_file_upload)


def test_s3_backup_upload_hook_exists():
    """#6: optional S3 backup upload after local pg_dump."""
    from backend.tasks import maintenance_tasks as mt

    src = inspect.getsource(mt)
    assert "_maybe_upload_backup_to_s3" in src
    assert "BACKUP_S3_BUCKET" in src
    assert "boto3" in src
    # Safe no-op without bucket
    assert mt._maybe_upload_backup_to_s3("/tmp/fake.sql.gz") is None


def test_canary_deploy_workflow_with_health_and_rollback():
    """#101: canary workflow with health gate + rollback path."""
    path = Path(".github/workflows/canary-deploy.yml")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Canary" in text or "canary" in text
    assert "health" in text.lower()
    assert "rollback" in text.lower()
    assert "workflow_dispatch" in text


def test_terraform_iac_scaffold_for_backups():
    """#105: Terraform scaffold for backup infrastructure."""
    root = Path("infrastructure/terraform")
    assert (root / "main.tf").exists()
    assert (root / "variables.tf").exists()
    assert (root / "outputs.tf").exists()
    main = (root / "main.tf").read_text(encoding="utf-8")
    assert "aws_s3_bucket" in main
    assert "backup" in main.lower()


def test_ssl_configuration_artifacts_present():
    """#2: SSL/TLS deployment artifacts present."""
    assert Path("infrastructure/docker/nginx/nginx-ssl.conf").exists()
    ssl_conf = Path("infrastructure/docker/nginx/nginx-ssl.conf").read_text(
        encoding="utf-8"
    )
    assert "ssl" in ssl_conf.lower()
    assert Path("docs/SSL_DEPLOYMENT_GUIDE.md").exists()


def test_staging_and_production_deploy_workflows():
    """#100 / #3: staging + production deploy workflows exist with build/health paths."""
    staging = Path(".github/workflows/staging-deploy.yml")
    prod = Path(".github/workflows/production-deploy.yml")
    assert staging.exists() and prod.exists()
    st = staging.read_text(encoding="utf-8")
    assert "build" in st.lower()
    assert "staging" in st.lower()
    pr = prod.read_text(encoding="utf-8")
    assert "release" in pr.lower() or "workflow_dispatch" in pr
    assert Path("docs/PRODUCTION_DEPLOYMENT_GUIDE.md").exists()
    assert Path("start.sh").exists()


def test_ml_training_artifacts_and_pipelines():
    """#37 / #47: training pipelines + model artifact metadata present."""
    assert Path("backend/ml/training/train_lstm.py").exists()
    assert Path("backend/ml/training/train_xgboost.py").exists()
    assert Path("backend/ml/training/train_prophet.py").exists()
    # Config / results committed even if large weight files are absent
    assert Path("ml_models/lstm_config.json").exists()
    assert Path("ml_models/xgboost_config.json").exists()
    assert Path("ml_models/lstm_training_results.json").exists()
    assert Path("ml_models/xgboost_training_results.json").exists()


def test_beta_onboarding_and_api_key_env_docs():
    """#32 / #30: beta onboarding guide + API key env documentation."""
    beta = Path("docs/BETA_ONBOARDING_GUIDE.md")
    assert beta.exists()
    text = beta.read_text(encoding="utf-8")
    assert "Onboarding" in text or "onboarding" in text.lower()
    assert "beta" in text.lower() or "Beta" in text

    env = Path(".env.example").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" in env or "ANTHROPIC_API_KEY" in env
