"""FIFO scheduler boundary."""
from rollout_engine.queue import RequestQueue
from rollout_engine.api import RolloutRequest

def next_request(queue: RequestQueue) -> RolloutRequest | None:
    return queue.pop() if len(queue) else None
