"""Worker protocol and deterministic local fake."""
from typing import Protocol
from rollout_engine.api import RolloutRequest, RolloutResult

class Worker(Protocol):
    async def generate(self, request: RolloutRequest) -> list[RolloutResult]: ...

class EchoWorker:
    def __init__(self, worker_id: str = "local") -> None:
        self.worker_id = worker_id

    async def generate(self, request: RolloutRequest) -> list[RolloutResult]:
        return [RolloutResult(request.id, request.prompt, request.model_version, self.worker_id)
                for _ in range(request.sample_count)]
