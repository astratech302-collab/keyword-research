"""Small bounded-concurrency helper used by independent API batches."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    workers: int,
    on_done: Callable[[int, int], None] | None = None,
) -> list[R]:
    """Map in parallel while returning results in input order.

    The worker count is deliberately bounded by configuration. If a worker fails,
    the exception is propagated so the pipeline checkpoint remains resumable.
    """
    total = len(items)
    if not total:
        return []
    if workers <= 1 or total == 1:
        out = []
        for done, item in enumerate(items, 1):
            out.append(fn(item))
            if on_done:
                on_done(done, total)
        return out

    out: list[R | None] = [None] * total
    pool = ThreadPoolExecutor(max_workers=min(workers, total), thread_name_prefix="kwresearch")
    futures = {}
    try:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for done, future in enumerate(as_completed(futures), 1):
            out[futures[future]] = future.result()
            if on_done:
                on_done(done, total)
    except BaseException:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    return out  # type: ignore[return-value]
