# Implementation plan

## Main boundary

The rollout engine should not own the training algorithm. Its job is to accept generation requests, produce verified samples, and return them with enough metadata for the trainer to compute updates.

## Core records

### RolloutRequest

- prompt / conversation
- model version
- sampling parameters
- number of samples
- verifier policy
- deadline / priority

### RolloutResult

- generated tokens/text
- log-probability metadata when requested
- reward / verifier output
- worker and model version
- timing information

## Scheduler

Start with FIFO and static batching. Then add:

1. size-aware batching
2. priority/deadline queueing
3. straggler-aware reassignment
4. GPU-memory-aware placement
5. throughput-aware policy

## Delivery order

1. local mock worker
2. request/result schema
3. Redis queue
4. Ray worker pool
5. vLLM adapter
6. verifier service
7. dynamic batching
8. metrics and benchmark harness
9. scheduling experiments

## Scaffold checkpoint

A deterministic local fake worker and request/result schema are in place. All external queue, GPU, and distributed milestones remain planned.
