"""Verifier interface; policy-specific scoring is planned."""
from typing import Protocol
from rollout_engine.api import RolloutResult

class Verifier(Protocol):
    async def score(self, result: RolloutResult) -> float: ...

class NonEmptyVerifier:
    async def score(self, result: RolloutResult) -> float:
        return float(bool(result.text.strip()))
