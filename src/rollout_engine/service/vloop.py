"""Clocks and event loops for the live service.

``VirtualTimeLoop`` is an asyncio event loop with a virtual clock: it never sleeps and never
touches a socket; when nothing is ready to run at the current instant, it first wakes the
tasks waiting in ``idle()`` (so the controller can process an instant only after every
message of that instant arrived) and otherwise jumps to the next timer. Time is kept as
integer milliseconds, so timers set at whole milliseconds fire exactly.

``ScaledClock`` runs the service on the wall clock, optionally accelerated (the demo).
"""

from __future__ import annotations

import asyncio
import selectors
import time


class _VirtualSelector(selectors.BaseSelector):
    def __init__(self):
        self._map: dict = {}
        self.loop: VirtualTimeLoop | None = None

    def register(self, fileobj, events, data=None):
        fd = fileobj if isinstance(fileobj, int) else fileobj.fileno()
        key = selectors.SelectorKey(fileobj, fd, events, data)
        self._map[fd] = key
        return key

    def unregister(self, fileobj):
        fd = fileobj if isinstance(fileobj, int) else fileobj.fileno()
        return self._map.pop(fd)

    def modify(self, fileobj, events, data=None):
        self.unregister(fileobj)
        return self.register(fileobj, events, data)

    def select(self, timeout=None):
        loop = self.loop
        if timeout is not None and timeout <= 0:
            return []
        if loop._idle_waiters:
            waiters, loop._idle_waiters = loop._idle_waiters, []
            for f in waiters:
                if not f.done():
                    f.set_result(None)
            return []
        if timeout is None:
            raise RuntimeError("virtual-time loop: nothing is scheduled (deadlock)")
        loop._now_ms = max(loop._now_ms, round((loop._now_ms / 1000 + timeout) * 1000))
        return []

    def get_map(self):
        return self._map

    def close(self):
        self._map.clear()


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    def __init__(self):
        sel = _VirtualSelector()
        self._now_ms = 0
        self._idle_waiters: list[asyncio.Future] = []
        super().__init__(selector=sel)
        sel.loop = self
        self._clock_resolution = 1e-9

    # no self-pipe: nothing is ever woken from another thread, so no socket is created
    def _make_self_pipe(self):
        self._ssock = self._csock = None

    def _close_self_pipe(self):
        pass

    def _write_to_self(self):
        pass

    def time(self) -> float:
        return self._now_ms / 1000

    def now_ms(self) -> int:
        return self._now_ms

    def idle(self) -> asyncio.Future:
        """A future resolved when nothing else can run at the current instant."""
        f = self.create_future()
        self._idle_waiters.append(f)
        return f


class LoopClock:
    """Milliseconds of a virtual-time loop (the core's clock in conformance tests)."""

    def __init__(self, loop: VirtualTimeLoop, offset_ms: int = 0):
        self.loop = loop
        self.offset = offset_ms  # a restarted controller continues the stored timeline
        self.virtual = True

    def now_ms(self) -> int:
        return self.offset + self.loop.now_ms()

    async def sleep_until(self, t_ms: int) -> None:
        delay = (t_ms - self.now_ms()) / 1000
        await asyncio.sleep(max(0.0, delay))

    async def settle(self) -> None:
        await self.loop.idle()


class ScaledClock:
    """Wall clock in integer ms since construction, ``speed`` times faster than real time."""

    def __init__(self, speed: float = 1.0, offset_ms: int = 0):
        self.speed = speed
        self.offset = offset_ms
        self.t0 = time.monotonic()
        self.virtual = False

    def now_ms(self) -> int:
        return self.offset + int((time.monotonic() - self.t0) * 1000 * self.speed)

    async def sleep_until(self, t_ms: int) -> None:
        while True:
            d = (t_ms - self.now_ms()) / 1000 / self.speed
            if d <= 0:
                return
            await asyncio.sleep(d)

    async def settle(self) -> None:
        # real time: let the messages that are already in flight arrive
        for _ in range(3):
            await asyncio.sleep(0)


def run_virtual(coro_factory):
    """Run ``coro_factory(loop)`` on a fresh virtual-time loop and return its result."""
    loop = VirtualTimeLoop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro_factory(loop))
    finally:
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        asyncio.set_event_loop(None)
        loop.close()
