"""Async retry with exponential backoff and jitter for network calls.

Provider SDKs retry on their own, each with different defaults and without
Jarvis' structured logs. Keeping one policy here means an outage looks the
same in the logs whether Anthropic, Ollama or a web tool failed, every client
agrees on what "transient" means, and tests can drive the whole thing with a
fake clock instead of waiting.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final

from jarvis.utils.logging import get_logger

SleepFn = Callable[[float], Awaitable[None]]
"""Something that waits for the given number of seconds."""

RandomFn = Callable[[], float]
"""Source of uniform random numbers in ``[0, 1)``, injectable for tests."""

log = get_logger(__name__)

_RNG: Final = random.SystemRandom()


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many times to retry and how long to wait between attempts.

    Delays grow geometrically from ``base_delay`` and are capped at
    ``max_delay``; each is then scaled by a random factor in
    ``[1 - jitter, 1 + jitter]`` so that clients that failed together do not
    retry in lockstep and hammer a recovering service.

    Attributes:
        max_attempts: Total attempts including the first; ``1`` disables retries.
        base_delay: Seconds before the first retry.
        max_delay: Upper bound for any single delay, before jitter.
        multiplier: Growth factor between consecutive delays.
        jitter: Fraction of the delay to randomise, in ``[0, 1]``.
    """

    max_attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 8.0
    multiplier: float = 2.0
    jitter: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if not 0.0 <= self.jitter <= 1.0:
            raise ValueError("jitter must be between 0 and 1")

    def delay_for(self, attempt: int, *, random_fraction: float) -> float:
        """Seconds to wait after failed attempt number ``attempt`` (1-based).

        Args:
            attempt: The attempt that just failed.
            random_fraction: Uniform random number in ``[0, 1)`` driving the
                jitter; injected so tests are deterministic.
        """
        capped = min(self.max_delay, self.base_delay * self.multiplier ** (attempt - 1))
        return capped * (1.0 + self.jitter * (2.0 * random_fraction - 1.0))


DEFAULT_RETRY_POLICY: Final = RetryPolicy()
"""Four attempts over roughly five seconds: long enough to ride out a blip, short
enough that a user waiting for an answer is not left wondering."""


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    is_retryable: Callable[[Exception], bool],
    policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    sleep: SleepFn = asyncio.sleep,
    random_fraction: RandomFn = _RNG.random,
    operation_name: str = "operation",
) -> T:
    """Run ``operation`` until it succeeds or the policy gives up.

    Only errors accepted by ``is_retryable`` are retried; anything else (a
    validation error, a bad API key) is raised immediately because repeating
    it cannot help. Retrying only wraps the call itself: for a streaming
    request that means establishing the stream, since output already handed
    to the caller cannot be replayed.

    Args:
        operation: Zero-argument coroutine factory to call on every attempt.
        is_retryable: Decides whether a failure is transient.
        policy: Attempt count and backoff schedule.
        sleep: Replaces ``asyncio.sleep`` in tests.
        random_fraction: Replaces the jitter source in tests.
        operation_name: Label for the log lines.

    Returns:
        Whatever ``operation`` returns on the first successful attempt.

    Raises:
        Exception: The last error, once attempts are exhausted or the error is
            not retryable.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return await operation()
        except Exception as error:
            if attempt >= policy.max_attempts or not is_retryable(error):
                raise
            delay = policy.delay_for(attempt, random_fraction=random_fraction())
            log.warning(
                "retry.scheduled",
                operation=operation_name,
                attempt=attempt,
                max_attempts=policy.max_attempts,
                delay_s=round(delay, 3),
                error=repr(error),
            )
            await sleep(delay)


# Design notes:
# - The retryable predicate is passed in rather than built from exception
#   types here: each client knows its own SDK's hierarchy, and utils must not
#   import provider packages.
# - asyncio.CancelledError derives from BaseException, so `except Exception`
#   never swallows a cancellation; barge-in relies on that.
# - SystemRandom only because bandit (S311) flags the default generator; the
#   jitter has no security role.
