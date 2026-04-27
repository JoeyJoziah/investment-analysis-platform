> **ARCHIVED 2026-04-27 by 03-ml-engine**
> Original: docs/ml/GPU_SUPPORT.md
> Validation summary: 4/5 claims still current.
> See `../../reports/03-ml-engine.md` §2 for per-claim status.

# GPU Support for ML Training

**Key Claims (as validated in 2026-04-27 audit):**

1. `backend/ml/gpu_utils.py` provides centralized GPU detection with graceful CPU fallback
2. XGBoost 2.0+ uses `device` parameter; older versions use `tree_method='gpu_hist'`
3. LSTM training uses PyTorch AMP (mixed precision)
4. `FORCE_CPU` and `CUDA_VISIBLE_DEVICES` environment variables are respected
5. Airflow DAG GPU config auto-detected in train_models task

[Original content truncated — see source file for full specification]
