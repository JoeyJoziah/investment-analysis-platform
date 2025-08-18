#!/bin/bash
echo "Starting Investment Analysis App..."
python3 -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
