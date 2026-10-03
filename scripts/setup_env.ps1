# Create .venv in the repository and install the package with its dev tools.
# Usage (repository root): powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
# Then use .venv\Scripts\python.exe, or activate the environment in your shell.
$ErrorActionPreference = "Stop"
python -m venv .venv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -m pip install --disable-pip-version-check -q -e ".[dev]"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .\.venv\Scripts\python.exe -c "import rollout_engine, numpy, scipy; print(rollout_engine.__version__, numpy.__version__, scipy.__version__)"
exit $LASTEXITCODE
