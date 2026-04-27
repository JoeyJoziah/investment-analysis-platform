> **ARCHIVED 2026-04-27 by 03-ml-engine**
> Original: docs/ml/ML_QUICKSTART.md
> Validation summary: 3/4 claims still current.
> See `../../reports/03-ml-engine.md` §2 for per-claim status.

# ML Quick Start Guide

**Version**: 1.0.0
**Last Updated**: August 19, 2025

**Key Claims (as validated in 2026-04-27 audit):**

1. ML services start via: `docker-compose up -d ml-api backend database redis`
2. ML API health at `http://localhost:8001/health`
3. First prediction via POST to `http://localhost:8001/predict` with features array
4. Repo claims "sample_model" is available on startup

[Original content truncated — see source file for full specification]
