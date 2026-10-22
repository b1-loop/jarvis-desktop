import subprocess
import sys

from jarvis.integrations.connect.common import (
    REDACTED,
    RedactingEcho,
    mask_email,
    mask_identifier,
)


def test_identifiers_keep_only_their_edges() -> None:
    assert mask_identifier("1234567890-abc.apps.googleusercontent.com") == "12****om"
    assert mask_identifier("abc") == "***"


def test_email_keeps_the_domain() -> None:
    assert mask_email("bojidar@gmail.com") == "bo****@gmail.com"
    assert mask_email("not-an-address") == "no****ss"


def test_redaction_hides_known_secrets_in_any_line() -> None:
    lines: list[str] = []
    echo = RedactingEcho(lines.append, ["fake-bot-token-123", None, "short"])
    echo.add_secret("fake-access-token-456")

    echo("error body: Bot fake-bot-token-123 / Bearer fake-access-token-456 / short")

    assert lines == [f"error body: Bot {REDACTED} / Bearer {REDACTED} / short"]


def test_importing_the_connect_cli_needs_no_auth_sdk() -> None:
    # `jarvis --help` imports the connect commands; the SDKs must stay lazy.
    code = (
        "import sys, jarvis.__main__; "
        "print([m for m in ('msal', 'google_auth_oauthlib') if m in sys.modules])"
    )
    result = subprocess.run(  # noqa: S603 - the test's own interpreter, fixed code
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "[]"
