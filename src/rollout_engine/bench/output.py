"""Deterministic CSV/JSON output and run manifests."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import platform
import subprocess
import sys

import numpy as np
import scipy

FLOAT_FMT = "{:.6g}"


def fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return FLOAT_FMT.format(v)
    return str(v)


def columns_of(rows: list[dict], first: list[str]) -> list[str]:
    cols = list(first)
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    return cols


def write_csv(path: str, rows: list[dict], first: list[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = columns_of(rows, first)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for r in rows:
        w.writerow([fmt(r.get(c)) for c in cols])
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(buf.getvalue())


def read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def git_commit() -> dict:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
        return {"commit": sha or "unknown", "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}


def cpu_model() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            k = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def ram_gb() -> float | None:
    if sys.platform == "win32":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return round(m.ullTotalPhys / 1e9, 1)
        return None
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) * 1024 / 1e9, 1)
    except OSError:
        pass
    return None


def hardware() -> dict:
    return {
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": ram_gb(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
    }


def software() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "highs": "1.2.0 (vendored by SciPy 1.13.1)"
        if scipy.__version__ == "1.13.1"
        else "vendored by SciPy " + scipy.__version__,
    }
