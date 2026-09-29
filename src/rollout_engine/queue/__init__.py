"""FIFO in-memory request queue."""
from collections import deque
from rollout_engine.api import RolloutRequest

class RequestQueue:
    def __init__(self) -> None:
        self._items: deque[RolloutRequest] = deque()

    def put(self, request: RolloutRequest) -> None:
        self._items.append(request)

    def pop(self) -> RolloutRequest:
        return self._items.popleft()

    def __len__(self) -> int:
        return len(self._items)
