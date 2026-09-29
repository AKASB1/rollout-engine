"""Local result store."""
from rollout_engine.api import RolloutResult

class ResultStore:
    def __init__(self) -> None:
        self._results: dict[str, list[RolloutResult]] = {}

    def add(self, result: RolloutResult) -> None:
        self._results.setdefault(result.request_id, []).append(result)

    def get(self, request_id: str) -> list[RolloutResult]:
        return list(self._results.get(request_id, []))
