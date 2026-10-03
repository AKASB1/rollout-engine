#!/usr/bin/env bash
# Create .venv in the repository and install the package with its dev tools.
# Usage (repository root): bash scripts/setup_env.sh
# Then activate it: source .venv/bin/activate   (Git Bash on Windows: source .venv/Scripts/activate)
set -euo pipefail
PY=python3
if ! python3 -c "import sys" >/dev/null 2>&1; then PY=python; fi
"$PY" -m venv .venv
if [ -x .venv/bin/python ]; then VPY=.venv/bin/python; else VPY=.venv/Scripts/python.exe; fi
"$VPY" -m pip install --disable-pip-version-check -q -e ".[dev]"
"$VPY" -c "import rollout_engine, numpy, scipy, matplotlib; print('ready:', rollout_engine.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)"
