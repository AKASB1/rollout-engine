"""Check 8: determinism of runs, traces, the parallel runner, and the aggregates."""

import os
import random
import subprocess
import sys

import pytest

from conftest import ROOT, subprocess_env
from rollout_engine.bench.experiments import build_cells, load_config
from rollout_engine.bench.runner import ensure_trace, jobs_for, run_job, run_jobs, trace_path


def _strip(row):
    return {k: v for k, v in row.items() if not k.startswith("wall_")}


@pytest.fixture(scope="module")
def cfg():
    old = os.getcwd()
    os.chdir(ROOT)
    try:
        yield load_config()
    finally:
        os.chdir(old)


def _cells(cfg, only):
    return build_cells(cfg, None, quick=True, only=only)


def test_same_seed_same_rows_different_seed_differs(cfg, tmp_path):
    cell = _cells(cfg, ["S3"])[1]
    jobs = jobs_for([cell], seeds_of=lambda c: [1, 2], trace_root=str(tmp_path))
    a = [run_job(j)[0] for j in jobs]
    b = [run_job(j)[0] for j in jobs]
    assert a == b
    by_seed = {r["seed"]: r for r in a if r["policy"] == "reference"}
    assert _strip(by_seed[1]) != _strip(by_seed[2])


def test_trace_bytes_do_not_depend_on_policies(cfg, tmp_path):
    cell = _cells(cfg, ["S2"])[0]
    p1 = ensure_trace(cell.generator, 3, str(tmp_path / "a"))
    jobs_all = jobs_for([cell], seeds_of=lambda c: [3], trace_root=str(tmp_path / "b"))
    jobs_one = jobs_for(
        [cell],
        seeds_of=lambda c: [3],
        policies_of=lambda c: c.policies[:1],
        trace_root=str(tmp_path / "c"),
    )
    with open(p1, "rb") as f:
        ref = f.read()
    for js in (jobs_all, jobs_one):
        with open(js[0]["trace_path"], "rb") as f:
            assert f.read() == ref
    assert os.path.basename(trace_path(cell.generator, 3)) == os.path.basename(p1)


@pytest.mark.slow
def test_serial_and_parallel_runner_agree_regardless_of_completion_order(cfg, tmp_path):
    cells = _cells(cfg, ["S2"])
    jobs = jobs_for(cells, seeds_of=lambda c: [1, 2], trace_root=str(tmp_path))
    serial, _, v1 = run_jobs(jobs, 1)
    shuffled = list(jobs)
    random.Random(5).shuffle(shuffled)
    parallel, _, v2 = run_jobs(shuffled, 3)
    assert serial == parallel and v1 == v2 == []


@pytest.mark.slow
def test_benchmark_files_identical_in_fresh_processes_serial_and_parallel(tmp_path):
    outs = []
    for k, workers in enumerate([1, 1, 3]):
        out = tmp_path / f"r{k}"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "rollout_engine.bench",
                "--quick",
                "--only",
                "S1,S2",
                "--workers",
                str(workers),
                "--out",
                str(out),
            ],
            cwd=ROOT,
            env=subprocess_env(),
            check=True,
            capture_output=True,
            timeout=600,
        )
        outs.append(out)
    for name in ("runs.csv", "aggregate.csv", "paired.csv"):
        data = [(o / name).read_bytes() for o in outs]
        assert data[0] == data[1] == data[2], name
