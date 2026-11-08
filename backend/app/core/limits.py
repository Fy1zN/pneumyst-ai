"""Per-user rate limits over a sliding one-hour window.

Counts live in this process's memory, which is right for a single server. Several server processes behind a load
balancer would each count separately and need a shared store such as Redis.
"""
from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable

from backend.app.core.errors import ApiError

WINDOW_S = 3600.0


class RateLimiter:
    def __init__(self, limits: dict[str, int], clock: Callable[[], float] = time.monotonic):
        self.limits = limits
        self.clock = clock
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, action: str, user_id: str) -> None:
        """Count one use, or raise 429 with Retry-After when the user has reached the limit for this action."""
        limit = self.limits.get(action, 0)
        if limit <= 0:
            return
        now = self.clock()
        with self._lock:
            hits = self._hits[(action, user_id)]
            while hits and hits[0] <= now - WINDOW_S:
                hits.popleft()
            if len(hits) >= limit:
                wait = math.ceil(hits[0] + WINDOW_S - now)
                raise ApiError(429, "RATE_LIMITED", f"limit of {limit} {action} requests per hour reached",
                               headers={"Retry-After": str(wait)})
            hits.append(now)