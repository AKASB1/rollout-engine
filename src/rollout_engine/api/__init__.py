"""Request and result records for local rollout."""
from dataclasses import dataclass

@dataclass(frozen=True)
class RolloutRequest:
    id: str
    prompt: str
    model_version: str
    sample_count: int = 1

    def __post_init__(self) -> None:
        if not self.id or not self.prompt or self.sample_count < 1:
            raise ValueError("id, prompt, and positive sample_count required")

@dataclass(frozen=True)
class RolloutResult:
    request_id: str
    text: str
    model_version: str
    worker_id: str
