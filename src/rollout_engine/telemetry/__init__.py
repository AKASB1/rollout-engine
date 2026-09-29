"""Throughput helper for local benchmark samples."""
def samples_per_second(count: int, elapsed_seconds: float) -> float:
    if count < 0 or elapsed_seconds <= 0:
        raise ValueError("nonnegative count and positive elapsed time required")
    return count / elapsed_seconds
