import json
from pathlib import Path
from typing import Any

import pytest

from jarvis.integrations.connect.common import ConnectError
from jarvis.integrations.connect.google import (
    ClientSecrets,
    explain_google_failure,
    validate_client_secrets,
    verify_google,
)
from jarvis.integrations.connect.probe import ProbeResult
from tests.integrations.conftest import Router

DESKTOP_CLIENT = {
    "installed": {
        "client_id": "1234567890-fake.apps.googleusercontent.com",
        "project_id": "jarvis-test",
        "client_secret": "fake-client-secret",
        "redirect_uris": ["http://localhost"],
    }
}


def write(tmp_path: Path, content: Any) -> Path:
    path = tmp_path / "google_client.json"
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
    return path


def test_a_desktop_client_is_accepted(tmp_path: Path) -> None:
    secrets = validate_client_secrets(write(tmp_path, DESKTOP_CLIENT))

    assert secrets == ClientSecrets("1234567890-fake.apps.googleusercontent.com", "jarvis-test")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ({"web": {"client_id": "x", "client_secret": "y"}}, "'Web application' OAuth client"),
        ({"type": "service_account", "private_key": "fake"}, "service-account key"),
        ({"installed": {"client_id": "x"}}, "lacks client_secret"),
        ({"something": "else"}, "not an OAuth client file"),
        ("[1, 2]", "not an OAuth client file"),
        ("{not json", "not readable JSON"),
    ],
)
def test_wrong_files_are_explained(tmp_path: Path, content: Any, message: str) -> None:
    with pytest.raises(ConnectError, match=message):
        validate_client_secrets(write(tmp_path, content))


def test_missing_setting_and_missing_file_point_to_the_readme(tmp_path: Path) -> None:
    with pytest.raises(ConnectError, match="GOOGLE_CLIENT_SECRETS_FILE is not set") as unset:
        validate_client_secrets(None)
    with pytest.raises(ConnectError, match="does not exist"):
        validate_client_secrets(tmp_path / "nope.json")

    assert "Connect your accounts" in str(unset.value)


def test_messages_never_include_the_client_secret(tmp_path: Path) -> None:
    broken = {"installed": {"client_secret": "fake-client-secret"}}

    with pytest.raises(ConnectError) as caught:
        validate_client_secrets(write(tmp_path, broken))

    assert "fake-client-secret" not in str(caught.value)


async def test_verification_reports_gmail_and_calendar(router: Router) -> None:
    router.add(
        "GET",
        "/gmail/v1/users/me/profile",
        {"emailAddress": "bojidar@example.com", "messagesTotal": 1234},
    )
    router.add("GET", "/calendar/v3/users/me/calendarList", {"items": [{}, {}]})
    lines: list[str] = []

    ok = await verify_google(router.client(), "fake-access-token", lines.append)

    assert ok
    assert lines == [
        "Gmail: OK, signed in as bo****@example.com (1,234 messages).",
        "Google Calendar: OK, 2 calendar(s) visible.",
    ]
    assert router.requests[0].headers["Authorization"] == "Bearer fake-access-token"


async def test_a_disabled_api_is_diagnosed(router: Router) -> None:
    router.add("GET", "/gmail/v1/users/me/profile", {"emailAddress": "a@b.c"})
    router.add(
        "GET",
        "/calendar/v3/users/me/calendarList",
        {"error": {"code": 403, "status": "PERMISSION_DENIED", "details": ["SERVICE_DISABLED"]}},
        status=403,
    )
    lines: list[str] = []

    ok = await verify_google(router.client(), "fake-access-token", lines.append)

    assert not ok
    assert "Google Calendar API is not enabled" in lines[1]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (ProbeResult(status=0, network_error="ConnectTimeout"), "Could not reach Google"),
        (ProbeResult(status=403, text="ACCESS_TOKEN_SCOPE_INSUFFICIENT"), "--force"),
        (ProbeResult(status=401, text="{}"), "rejected the new token"),
        (ProbeResult(status=500, text="oops"), "HTTP 500"),
    ],
)
def test_failure_explanations(result: ProbeResult, expected: str) -> None:
    assert expected in explain_google_failure("Gmail API", result)
