# RL Post-Training Rollout Engine

A distributed rollout service for RLHF, GRPO, and verifier-based post-training. The project focuses on the systems side of sample generation: queueing, batching, GPU workers, verification, and backpressure.

**Status:** implementation scaffold.

## Scope

- rollout request queue
- asynchronous sampling workers
- dynamic batching by model and generation parameters
- GPU worker registration and health checks
- verifier / reward service integration
- rollout storage and replay metadata
- straggler detection and retry
- throughput, latency, and GPU utilization metrics
- scheduling policies for mixed rollout workloads

## Proposed stack

Python 3.12 · PyTorch · Ray · vLLM/SGLang adapters · Redis · gRPC · Prometheus

## Data path

```text
Trainer
  │
  ▼
Rollout Queue
  │
  ▼
Batcher / Scheduler
  │
  ├──► GPU Worker 1 ──► Verifier
  ├──► GPU Worker 2 ──► Verifier
  └──► GPU Worker N ──► Verifier
             │
             ▼
       Rollout Store
             │
             ▼
           Trainer
```

## Repository layout

```text
src/rollout_engine/
  api/
  queue/
  scheduler/
  batching/
  workers/
  verifier/
  storage/
  telemetry/
tests/
benchmarks/
configs/
```

See [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Reference projects

- [verl-project/verl](https://github.com/verl-project/verl) — RL post-training infrastructure and rollout/trainer integration
- [OpenRLHF/OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — distributed RLHF training stack
- [huggingface/trl](https://github.com/huggingface/trl) — post-training algorithms and trainer interfaces
- [vllm-project/vllm](https://github.com/vllm-project/vllm) — high-throughput LLM inference

## License

MIT
