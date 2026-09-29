"""Static compatible-request batching."""
from rollout_engine.api import RolloutRequest

def batch_by_model(requests: list[RolloutRequest]) -> dict[str, list[RolloutRequest]]:
    batches: dict[str, list[RolloutRequest]] = {}
    for request in requests:
        batches.setdefault(request.model_version, []).append(request)
    return batches
