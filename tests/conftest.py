import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def subprocess_env() -> dict:
    """Environment for child Python processes: the package importable without installation."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    return env
