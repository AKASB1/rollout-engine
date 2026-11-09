# Contributing

This is a portfolio repository with one maintainer. Issues and pull requests are welcome; this page lists what a change needs before it is merged.

## Environment

- Python 3.12 (`pyproject.toml`). `scripts/setup_env.ps1` (PowerShell) or `bash scripts/setup_env.sh` creates `.venv` and installs the package with the `dev` extra (pytest, pytest-repeat, ruff, matplotlib).
- No GPU, model, or inference engine is needed: the tests, the benchmark, and the demo run on the simulator and on mock workers.

## Checks

Run these before opening a pull request. CI runs the same steps (`.github/workflows/ci.yml`):

```bash
ruff check src tests scripts
ruff format --check src tests scripts
python -m pytest -q
python -m rollout_engine.bench --quick --workers 2 --out benchmarks/outputs/ci-quick
python -m rollout_engine.service.demo --port 0
```

Tests marked `slow` are part of the default run.

## Results

The results in the README and on the [results page](benchmarks/README.md) come from the committed files in `benchmarks/results/`, each run with its manifest; `benchmarks/results/full/tables.md` is generated from the CSV files. Policy parameters are tuned on the tuning seeds and frozen in `configs/tuned/` before an evaluation; a full evaluation refuses a `tuned.json` made for another experiment configuration. A change that moves a result reruns the tuning if the configuration changed, then the evaluation, once on the committed change, and commits the result files and manifests. The README's highlights and table are written by hand from the results.

## Figures and diagrams

`python scripts/plot_results.py` draws the figures in `docs/figures/` from the committed results, including the README version of S2, and `python scripts/plot_timeline.py` draws the timeline. The architecture diagram is drawn from `docs/diagrams/architecture.d2` with [D2](https://d2lang.com) (`bash scripts/render_diagrams.sh` or `powershell -ExecutionPolicy Bypass -File scripts\render_diagrams.ps1`); the committed SVG was drawn with D2 0.9.0. `python scripts/make_assets.py` writes the icon. CI needs none of these.

## Style

- Policies see only the read-only view of the scheduler: never the true length of an unfinished sample ([docs/contracts.md](docs/contracts.md)).
- The simulator and the live controller drive the same core state machine; a change to one driver keeps check 9 (the service reproduces the simulator on a virtual-time loop) passing.
- Every result is simulated with assumed parameters; documentation says so, names the assumption, and names the result file.
