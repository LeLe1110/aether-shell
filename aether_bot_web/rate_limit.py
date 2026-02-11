"""In-memory token bucket rate limiter."""

import time


class RateLimiter:
    """Per-user token bucket rate limiter."""

    def __init__(self, rpm: int = 20):
        self._rpm = rpm
        self._interval = 60.0 / rpm if rpm > 0 else 0
        self._buckets: dict[str, tuple[float, float]] = {}  # user_id -> (tokens, last_refill)
        self._last_cleanup = time.monotonic()

    def check(self, user_id: str) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        if self._rpm <= 0:
            return True

        now = time.monotonic()
        self._maybe_cleanup(now)

        if user_id in self._buckets:
            tokens, last_refill = self._buckets[user_id]
            elapsed = now - last_refill
            tokens = min(self._rpm, tokens + elapsed / self._interval)
            last_refill = now
        else:
            tokens = float(self._rpm)
            last_refill = now

        if tokens >= 1.0:
            self._buckets[user_id] = (tokens - 1.0, last_refill)
            return True

        self._buckets[user_id] = (tokens, last_refill)
        return False

    def _maybe_cleanup(self, now: float) -> None:
        """Remove stale buckets every 5 minutes."""
        if now - self._last_cleanup < 300:
            return
        self._last_cleanup = now
        stale = [uid for uid, (_, lr) in self._buckets.items() if now - lr > 600]
        for uid in stale:
            del self._buckets[uid]
