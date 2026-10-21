from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from jarvis.utils.retry import DEFAULT_RETRY_POLICY, RetryPolicy, retry_async


class TransientError(Exception):
    pass


class FatalError(Exception):
    pass


@dataclass
class FakeClock:
    """Records sleeps instead of waiting."""

    sleeps: list[float] = field(default_factory=list)

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def failing_then(outcomes: list[Exception | str]) -> Callable[[], object]:
    """Operation that pops one scripted outcome per call: raise it or return it."""

    async def operation() -> str:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return operation


def is_transient(error: Exception) -> bool:
    return isinstance(error, TransientError)


def no_jitter() -> float:
    return 0.5  # the midpoint of the jitter range leaves the delay unchanged


async def test_returns_first_success_without_sleeping() -> None:
    clock = FakeClock()

    result = await retry_async(failing_then(["ok"]), is_retryable=is_transient, sleep=clock.sleep)

    assert result == "ok"
    assert clock.sleeps == []


async def test_retries_transient_errors_with_exponential_backoff() -> None:
    clock = FakeClock()
    policy = RetryPolicy(max_attempts=4, base_delay=0.5, multiplier=2.0, max_delay=8.0)

    result = await retry_async(
        failing_then([TransientError(), TransientError(), "ok"]),
        is_retryable=is_transient,
        policy=policy,
        sleep=clock.sleep,
        random_fraction=no_jitter,
    )

    assert result == "ok"
    assert clock.sleeps == [0.5, 1.0]


async def test_gives_up_after_max_attempts_and_raises_the_last_error() -> None:
    clock = FakeClock()
    policy = RetryPolicy(max_attempts=3, base_delay=1.0)

    with pytest.raises(TransientError):
        await retry_async(
            failing_then([TransientError(), TransientError(), TransientError("last")]),
            is_retryable=is_transient,
            policy=policy,
            sleep=clock.sleep,
            random_fraction=no_jitter,
        )

    assert clock.sleeps == [1.0, 2.0]


async def test_non_retryable_errors_are_raised_immediately() -> None:
    clock = FakeClock()

    with pytest.raises(FatalError):
        await retry_async(
            failing_then([FatalError(), "never reached"]),
            is_retryable=is_transient,
            sleep=clock.sleep,
        )

    assert clock.sleeps == []


async def test_delay_is_capped_at_max_delay() -> None:
    clock = FakeClock()
    policy = RetryPolicy(max_attempts=5, base_delay=1.0, multiplier=10.0, max_delay=3.0)

    await retry_async(
        failing_then([TransientError(), TransientError(), TransientError(), "ok"]),
        is_retryable=is_transient,
        policy=policy,
        sleep=clock.sleep,
        random_fraction=no_jitter,
    )

    assert clock.sleeps == [1.0, 3.0, 3.0]


def test_jitter_scales_the_delay_symmetrically() -> None:
    policy = RetryPolicy(base_delay=2.0, jitter=0.25)

    assert policy.delay_for(1, random_fraction=0.0) == pytest.approx(1.5)
    assert policy.delay_for(1, random_fraction=0.5) == pytest.approx(2.0)
    assert policy.delay_for(1, random_fraction=1.0) == pytest.approx(2.5)


def test_policy_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="jitter"):
        RetryPolicy(jitter=1.5)


def test_default_policy_finishes_within_a_few_seconds() -> None:
    worst_case = sum(
        DEFAULT_RETRY_POLICY.delay_for(attempt, random_fraction=1.0)
        for attempt in range(1, DEFAULT_RETRY_POLICY.max_attempts)
    )

    assert worst_case < 10.0
