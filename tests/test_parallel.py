import threading
import time

import pytest

from kwresearch.parallel import parallel_map


def test_parallel_map_is_bounded_and_keeps_input_order():
    lock = threading.Lock()
    active = 0
    peak = 0

    def work(value):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return value * 2

    assert parallel_map(work, list(range(12)), workers=3) == [i * 2 for i in range(12)]
    assert 1 < peak <= 3


def test_parallel_map_propagates_worker_failure():
    def work(value):
        if value == 2:
            raise RuntimeError("rate limit")
        time.sleep(0.01)
        return value

    with pytest.raises(RuntimeError, match="rate limit"):
        parallel_map(work, list(range(20)), workers=3)
