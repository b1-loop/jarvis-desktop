from collections.abc import Callable

import pytest

from jarvis.config import Settings
from jarvis.integrations.connect.common import ConnectError
from jarvis.integrations.connect.microsoft import (
    TerminalDeviceCodePrompt,
    describe_plan,
    explain_graph_failure,
    render_device_code,
    verify_microsoft,
    wants_teams,
)
from jarvis.integrations.connect.probe import ProbeResult
from jarvis.integrations.microsoft_graph import DeviceCode
from tests.integrations.conftest import Router

CODE = DeviceCode(
    user_code="ABCD-1234",
    verification_uri="https://microsoft.com/devicelogin",
    message="To sign in, use a web browser...",
    expires_in=900,
)


def test_device_code_is_rendered_prominently() -> None:
    lines = render_device_code(CODE, copied=True, opened=True)

    assert lines[0] == lines[-1] == "=" * 64
    assert any(
        "https://microsoft.com/devicelogin" in line and "opened in your browser" in line
        for line in lines
    )
    assert any("ABCD-1234" in line and "copied to the clipboard" in line for line in lines)
    assert any("expires in 15 minutes" in line for line in lines)


def test_without_clipboard_or_browser_the_notes_are_left_out() -> None:
    lines = render_device_code(
        DeviceCode("ABCD-1234", "https://x.invalid", "m"), copied=False, opened=False
    )

    assert not any("clipboard" in line or "browser" in line for line in lines)
    assert not any("expires" in line for line in lines)


def test_prompt_copies_the_code_and_opens_the_page() -> None:
    copied: list[str] = []
    opened: list[str] = []
    lines: list[str] = []

    def copy(text: str) -> bool:
        copied.append(text)
        return True

    def open_url(url: str) -> bool:
        opened.append(url)
        return False

    TerminalDeviceCodePrompt(lines.append, copy=copy, open_url=open_url)(CODE)

    assert copied == ["ABCD-1234"]
    assert opened == ["https://microsoft.com/devicelogin"]
    assert not any("opened in your browser" in line for line in lines)


def test_teams_follows_the_flag_then_the_setting(make_settings: Callable[..., Settings]) -> None:
    off, on = make_settings(), make_settings(teams_enabled=True)

    assert wants_teams(off, None) is False
    assert wants_teams(on, None) is True
    assert wants_teams(off, True) is True
    assert wants_teams(on, False) is False


def test_plan_explains_teams_and_the_setting_mismatch(
    make_settings: Callable[..., Settings],
) -> None:
    lines: list[str] = []

    describe_plan(make_settings(ms_client_id="00000000-fake-id"), teams=True, echo=lines.append)

    text = "\n".join(lines)
    assert "00****id" in text and "00000000-fake-id" not in text
    assert "Chat.Read" in text
    assert "personal Microsoft accounts" in text
    assert "JARVIS_TEAMS_ENABLED=true" in text


def test_plan_without_teams_asks_for_outlook_only(make_settings: Callable[..., Settings]) -> None:
    lines: list[str] = []

    describe_plan(make_settings(ms_client_id="fake-client-id"), teams=False, echo=lines.append)

    assert "Mail.ReadWrite, Mail.Send, Calendars.ReadWrite" in lines[1]
    assert not any("Chat.Read" in line for line in lines)


def test_plan_requires_the_client_id(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(ConnectError, match="MS_CLIENT_ID is not set"):
        describe_plan(make_settings(), teams=False, echo=lambda _line: None)


async def test_verification_checks_mail_and_calendar(router: Router) -> None:
    router.add("GET", "/v1.0/me/mailFolders/inbox", {"totalItemCount": 1500, "unreadItemCount": 3})
    router.add("GET", "/v1.0/me/calendars", {"value": [{"name": "Calendar"}]})
    lines: list[str] = []

    ok = await verify_microsoft(
        router.client(), "fake-token", lines.append, username="me@outlook.com", teams=False
    )

    assert ok
    assert lines == [
        "Signed in as me****@outlook.com.",
        "Outlook mail: OK, inbox has 1,500 messages (3 unread).",
        "Outlook calendar: OK, 1 calendar(s) visible.",
    ]
    assert all(r.url.path != "/v1.0/me/chats" for r in router.requests)


async def test_teams_failure_on_a_personal_account_is_explained(router: Router) -> None:
    router.add("GET", "/v1.0/me/mailFolders/inbox", {"totalItemCount": 1, "unreadItemCount": 0})
    router.add("GET", "/v1.0/me/calendars", {"value": []})
    router.add("GET", "/v1.0/me/chats", {"error": {"code": "Forbidden"}}, status=403)
    lines: list[str] = []

    ok = await verify_microsoft(router.client(), "fake", lines.append, username=None, teams=True)

    assert not ok
    assert lines[-1].startswith("Teams chats: FAILED.")
    assert "personal Microsoft accounts" in lines[-1]


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (ProbeResult(status=0, network_error="ConnectError"), "Could not reach Microsoft Graph"),
        (
            ProbeResult(status=404, body={"error": {"code": "MailboxNotEnabledForRESTAPI"}}),
            "no mailbox",
        ),
        (ProbeResult(status=401, body={}), "--force"),
        (ProbeResult(status=403, body={"error": {"code": "ErrorAccessDenied"}}), "administrator"),
        (ProbeResult(status=500, body={"error": {"code": "Boom"}}), "HTTP 500 Boom."),
    ],
)
def test_graph_failure_explanations(result: ProbeResult, expected: str) -> None:
    assert expected in explain_graph_failure("Outlook mail", result)
