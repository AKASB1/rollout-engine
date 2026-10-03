"""Per-component random streams (conventions of llm-serving-control contracts v1, section 1).

Every component draws from its own stream derived from the run seed and the component
name, so adding or removing a component never shifts another component's numbers.

    c  = fnv1a64(name)
    s1 = splitmix64(seed ^ c)
    s2 = splitmix64(s1 ^ 0x9e3779b97f4a7c15 ^ c)
    stream = numpy PCG64DXSM with 128-bit state (s1 << 64) | s2 and a fixed odd increment

The derivation of (s1, s2) is the one of the Go projects; the generator itself differs, so
the streams are not bit-identical to theirs. The ``random`` module and ``hash()`` are never
used for anything that reaches a result.
"""

from __future__ import annotations

import numpy as np

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
# Fixed odd increment of the PCG stream (the 128-bit increment of Go's PCG, made odd).
PCG_INC = ((6364136223846793005 << 64) | 1442695040888963407) | 1


def fnv1a64(text: str) -> int:
    """64-bit FNV-1a over the UTF-8 bytes of ``text``."""
    h = FNV_OFFSET
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * FNV_PRIME) & MASK64
    return h


def splitmix64(x: int) -> int:
    """The SplitMix64 output function applied to ``x + golden`` (one SplitMix64 step)."""
    x = (x + GOLDEN) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def stream_seeds(seed: int, name: str) -> tuple[int, int]:
    c = fnv1a64(name)
    s1 = splitmix64((seed & MASK64) ^ c)
    s2 = splitmix64(s1 ^ GOLDEN ^ c)
    return s1, s2


def stream(seed: int, name: str) -> np.random.Generator:
    """A fresh, deterministic generator for component ``name`` of run ``seed``."""
    s1, s2 = stream_seeds(seed, name)
    bg = np.random.PCG64DXSM()
    bg.state = {
        "bit_generator": "PCG64DXSM",
        "state": {"state": (s1 << 64) | s2, "inc": PCG_INC},
        "has_uint32": 0,
        "uinteger": 0,
    }
    return np.random.Generator(bg)
