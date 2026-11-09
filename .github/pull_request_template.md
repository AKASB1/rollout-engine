## What this change does

## Checklist

- [ ] `ruff check src tests scripts` and `ruff format --check src tests scripts` are clean
- [ ] `python -m pytest -q` passes
- [ ] `python -m rollout_engine.bench --quick --workers 2 --out benchmarks/outputs/ci-quick` and `python -m rollout_engine.service.demo --port 0` run
- [ ] Tests added or updated for the changed behavior
- [ ] If the change moves a result: tuning and evaluation rerun as needed, result files and manifests committed
- [ ] Documentation updated; every new number names its result file
