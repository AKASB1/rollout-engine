import asyncio
import unittest
from rollout_engine.api import RolloutRequest
from rollout_engine.queue import RequestQueue
from rollout_engine.scheduler import next_request
from rollout_engine.workers import EchoWorker

class RolloutTests(unittest.TestCase):
    def test_fifo_and_samples(self):
        queue = RequestQueue()
        queue.put(RolloutRequest("one", "hello", "v1", 2))
        request = next_request(queue)
        self.assertEqual(len(asyncio.run(EchoWorker().generate(request))), 2)
        self.assertIsNone(next_request(queue))
