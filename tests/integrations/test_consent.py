import pytest

from jarvis.integrations.base import AuthError, ConsentRequiredError
from jarvis.integrations.consent import (
    ConnectionLog,
    NotifyOnConsent,
    connect_command,
    consent_required,
)
from jarvis.integrations.tokens import MemoryTokenStore


class FailingSource:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    async def access_token(self) -> str:
        self.calls += 1
        raise self.error


def test_the_remedy_names_the_exact_connect_command() -> None:
    error = consent_required("google", "no stored token")

    assert connect_command("microsoft") == "uv run python scripts/connect_accounts.py microsoft"
    assert error.remedy == (
        "Google isn't connected yet; run "
        "`uv run python scripts/connect_accounts.py google` in a terminal."
    )
    assert str(error) == f"google: {error.remedy}"
    assert isinstance(error, AuthError)


def test_an_expired_sign_in_is_worded_differently() -> None:
    error = consent_required("microsoft", "invalid_grant", expired=True)

    assert error.remedy.startswith("The Microsoft sign-in has expired; run `uv run python")
    assert error.reason == "invalid_grant"


def test_connection_log_records_clears_and_reads_back() -> None:
    store = MemoryTokenStore()
    log = ConnectionLog(store)

    assert log.last("google") is None
    log.record("google", "Google isn't connected yet")
    message, at = log.last("google") or ("", "")
    log.clear("google")

    assert message == "Google isn't connected yet"
    assert at.endswith("+00:00")
    assert log.last("google") is None
    assert "google-error" not in store.data


async def test_notify_on_consent_reports_once_and_keeps_raising() -> None:
    error = consent_required("google", "no stored token")
    inner = FailingSource(error)
    seen: list[ConsentRequiredError] = []

    async def notify(reported: ConsentRequiredError) -> None:
        seen.append(reported)

    source = NotifyOnConsent(inner, notify)
    for _ in range(3):
        with pytest.raises(ConsentRequiredError):
            await source.access_token()

    assert seen == [error]
    assert inner.calls == 3


async def test_other_auth_errors_pass_through_without_a_notification() -> None:
    seen: list[ConsentRequiredError] = []

    async def notify(reported: ConsentRequiredError) -> None:
        seen.append(reported)

    source = NotifyOnConsent(FailingSource(AuthError("google", "HTTP 403")), notify)
    with pytest.raises(AuthError, match="HTTP 403"):
        await source.access_token()

    assert seen == []
