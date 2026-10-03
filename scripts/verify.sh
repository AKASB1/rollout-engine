#!/usr/bin/env bash
# One-command build and test: unit and integration tests, the quick benchmark, the demo.
# Usage (repository root): bash scripts/verify.sh
# Interpreter: $PYTHON if set, else the active environment's python (or .venv's), else python.
set -euo pipefail
export PYTHONUTF8=1
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -z "${VIRTUAL_ENV:-}" ] && [ -x .venv/bin/python ]; then PY=.venv/bin/python
  elif [ -z "${VIRTUAL_ENV:-}" ] && [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
  else PY=python; fi
fi
"$PY" -m pytest -q
"$PY" -m rollout_engine.bench --quick --workers "${BENCH_WORKERS:-2}" --out benchmarks/outputs/verify-quick
"$PY" -m rollout_engine.service.demo --port 0
