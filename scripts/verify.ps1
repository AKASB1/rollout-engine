# One-command build and test: unit and integration tests, the quick benchmark, the demo.
# Usage (repository root): powershell -ExecutionPolicy Bypass -File scripts\verify.ps1
# Interpreter: $env:PYTHON if set, else .venv\Scripts\python.exe if it exists, else python on PATH.
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$py = "python"
if ($env:PYTHON) { $py = $env:PYTHON } elseif (Test-Path ".venv\Scripts\python.exe") { $py = ".venv\Scripts\python.exe" }
$workers = if ($env:BENCH_WORKERS) { $env:BENCH_WORKERS } else { "2" }
& $py -m pytest -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -m rollout_engine.bench --quick --workers $workers --out benchmarks/outputs/verify-quick
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -m rollout_engine.service.demo --port 0
exit $LASTEXITCODE
