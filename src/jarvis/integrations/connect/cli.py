"""``connect_accounts.py`` / ``jarvis connect``: sign accounts in once, in a terminal.

Commands:
    google        Validate the client JSON, consent in the browser, verify.
    microsoft     Device-code sign-in (``--teams`` adds Teams chats), verify.
    discord       Diagnose the bot token, server membership and permissions.
    status        Configured / connected / expiry / last error, no secrets.
    disconnect    Delete one service's stored tokens.

Exit codes: 0 when everything worked, 1 when signing in worked but a check
failed, 2 for a setup problem (the message says what to change), 130 after
Ctrl+C. Every side effect sits in :class:`ConnectDeps`, which tests pass as
the Typer context object, so no test opens a browser or the network.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Final

import httpx
import typer

from jarvis.config import Settings
from jarvis.integrations.base import IntegrationError
from jarvis.integrations.connect import discord, google, microsoft
from jarvis.integrations.connect.common import BlockingSignIn, ConnectError, Echo, RedactingEcho
from jarvis.integrations.connect.status import DISCORD_CHECK_KEY, collect_status, render_status
from jarvis.integrations.consent import ERROR_KEY_SUFFIX, SERVICE_LABELS, ConnectionLog
from jarvis.integrations.factory import TOKENS_SUBDIR
from jarvis.integrations.tokens import FileTokenStore, TokenStore

EXIT_OK: Final = 0
EXIT_CHECK_FAILED: Final = 1
EXIT_SETUP_PROBLEM: Final = 2
EXIT_INTERRUPTED: Final = 130
HTTP_TIMEOUT: Final = httpx.Timeout(20.0, connect=10.0)
RESTART_HINT: Final = "If Jarvis is running, quit it from the tray and start it again."

REVOKE_HINTS: Final[dict[str, str]] = {
    "google": "To revoke Jarvis's access on Google's side too: "
    "https://myaccount.google.com/permissions",
    "microsoft": "To revoke on Microsoft's side too: https://account.live.com/consent/Manage "
    "(personal accounts) or https://myapplications.microsoft.com (work accounts).",
    "discord": "Discord uses DISCORD_BOT_TOKEN from .env; remove it there to stop using the bot.",
}


class Service(StrEnum):
    """Services with stored state that ``disconnect`` can delete."""

    GOOGLE = "google"
    MICROSOFT = "microsoft"
    DISCORD = "discord"


STORE_KEYS: Final[dict[Service, tuple[str, ...]]] = {
    Service.GOOGLE: ("google", "google" + ERROR_KEY_SUFFIX),
    Service.MICROSOFT: ("microsoft", "microsoft" + ERROR_KEY_SUFFIX),
    Service.DISCORD: (DISCORD_CHECK_KEY, "discord" + ERROR_KEY_SUFFIX),
}


def default_store(settings: Settings) -> TokenStore:
    """The same store the app reads: ``<data_dir>/tokens``."""
    return FileTokenStore(settings.data_dir / TOKENS_SUBDIR)


def default_http() -> httpx.AsyncClient:
    """A short-lived client for the verification calls."""
    return httpx.AsyncClient(timeout=HTTP_TIMEOUT)


def default_microsoft_sign_in(
    settings: Settings, store: TokenStore, teams: bool, echo: Echo
) -> microsoft.MicrosoftSignIn:
    """The real msal sign-in with the terminal device-code prompt."""
    return microsoft.build_sign_in(settings, store, teams=teams, echo=echo)


@dataclass(frozen=True, slots=True)
class ConnectDeps:
    """Every effect of the connect commands; tests replace them with fakes."""

    settings: Callable[[], Settings] = Settings
    store: Callable[[Settings], TokenStore] = default_store
    http: Callable[[], httpx.AsyncClient] = default_http
    google_sign_in: Callable[[Settings, TokenStore], BlockingSignIn] = google.build_sign_in
    microsoft_sign_in: Callable[[Settings, TokenStore, bool, Echo], microsoft.MicrosoftSignIn] = (
        default_microsoft_sign_in
    )
    echo: Echo = typer.echo


@dataclass(slots=True)
class _Session:
    """What one command works with, built from the deps."""

    deps: ConnectDeps
    settings: Settings
    store: TokenStore
    echo: RedactingEcho

    def verify(self, check: Callable[[httpx.AsyncClient], Awaitable[bool]]) -> bool:
        async def run() -> bool:
            async with self.deps.http() as http:
                return await check(http)

        return asyncio.run(run())

    def finish(self, service: str, *, ok: bool, failure: str | None = None) -> None:
        """Record the outcome for ``status`` and exit 1 when a check failed."""
        label = SERVICE_LABELS[service]
        log = ConnectionLog(self.store)
        if ok:
            log.clear(service)
            self.echo(f"{label} is connected; the desktop app will use it silently.")
            self.echo(RESTART_HINT)
            return
        log.record(service, failure or f"{label} signed in but a check failed; run connect again.")
        raise typer.Exit(EXIT_CHECK_FAILED)


connect_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode=None,
    help="Connect Google, Microsoft and Discord once; the desktop app then signs in silently.",
)


def _session(ctx: typer.Context) -> _Session:
    deps = ctx.obj if isinstance(ctx.obj, ConnectDeps) else ConnectDeps()
    settings = deps.settings()
    token = settings.discord_bot_token
    echo = RedactingEcho(deps.echo, [token.get_secret_value() if token else None])
    return _Session(deps, settings, deps.store(settings), echo)


def _guarded(work: Callable[[], None], echo: Echo) -> None:
    """Map setup problems and Ctrl+C to messages and exit codes."""
    try:
        work()
    except (ConnectError, IntegrationError) as error:
        echo(f"Not connected: {error}")
        raise typer.Exit(EXIT_SETUP_PROBLEM) from error
    except KeyboardInterrupt:
        echo("Cancelled; nothing was changed.")
        raise typer.Exit(EXIT_INTERRUPTED) from None


@connect_app.command("google")
def connect_google(
    ctx: typer.Context,
    force: Annotated[
        bool, typer.Option("--force", help="Forget the stored sign-in first.")
    ] = False,
) -> None:
    """Gmail + Google Calendar: browser consent, then one check each."""
    session = _session(ctx)

    def work() -> None:
        secrets = google.validate_client_secrets(session.settings.google_client_secrets_file)
        google.describe_plan(session.settings, secrets, session.echo)
        if force:
            _forget(session.store, Service.GOOGLE)
        token = session.deps.google_sign_in(session.settings, session.store).acquire_blocking()
        session.echo.add_secret(token)
        session.echo("Signed in; the token is stored in the data folder (git-ignored).")
        ok = session.verify(lambda http: google.verify_google(http, token, session.echo))
        session.finish("google", ok=ok)

    _guarded(work, session.echo)


@connect_app.command("microsoft")
def connect_microsoft(
    ctx: typer.Context,
    teams: Annotated[
        bool | None,
        typer.Option(
            "--teams/--no-teams",
            help="Also request Teams chat access (work or school accounts only). "
            "Default: JARVIS_TEAMS_ENABLED.",
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Forget the stored sign-in first.")
    ] = False,
) -> None:
    """Outlook mail + calendar (and Teams with --teams): device-code sign-in, then checks."""
    session = _session(ctx)

    def work() -> None:
        with_teams = microsoft.wants_teams(session.settings, teams)
        microsoft.describe_plan(session.settings, teams=with_teams, echo=session.echo)
        if force:
            _forget(session.store, Service.MICROSOFT)
        auth = session.deps.microsoft_sign_in(
            session.settings, session.store, with_teams, session.echo
        )
        token = auth.acquire_blocking()
        session.echo.add_secret(token)
        username = auth.signed_in_username()
        ok = session.verify(
            lambda http: microsoft.verify_microsoft(
                http, token, session.echo, username=username, teams=with_teams
            )
        )
        session.finish("microsoft", ok=ok)

    _guarded(work, session.echo)


@connect_app.command("discord")
def connect_discord(ctx: typer.Context) -> None:
    """Check DISCORD_BOT_TOKEN and DISCORD_GUILD_ID against the Discord API."""
    session = _session(ctx)
    settings = session.settings
    check = asyncio.run(_diagnose(session.deps, settings))
    for finding in check.findings:
        mark = "OK  " if finding.ok else "FAIL"
        session.echo(f"[{mark}] {finding.title}: {finding.detail}")
    if check.ok:
        at = datetime.now(UTC).isoformat(timespec="seconds")
        record = {"at": at, "bot": check.bot_name, "guild": check.guild_name}
        session.store.save(DISCORD_CHECK_KEY, record)
        session.finish("discord", ok=True)
        return
    session.store.clear(DISCORD_CHECK_KEY)
    failed = next(f for f in check.findings if not f.ok)
    session.finish("discord", ok=False, failure=f"{failed.title}: {failed.detail}")


async def _diagnose(deps: ConnectDeps, settings: Settings) -> discord.DiscordCheck:
    async with deps.http() as http:
        return await discord.diagnose(http, settings.discord_bot_token, settings.discord_guild_id)


@connect_app.command("status")
def connect_status(ctx: typer.Context) -> None:
    """What is configured and connected; no secrets are printed."""
    session = _session(ctx)
    for line in render_status(collect_status(session.settings, session.store)):
        session.echo(line)


@connect_app.command("disconnect")
def disconnect(
    ctx: typer.Context,
    service: Annotated[Service, typer.Argument(case_sensitive=False, help="Which account.")],
) -> None:
    """Delete one service's stored tokens (the next use asks to connect again)."""
    session = _session(ctx)
    _forget(session.store, service)
    session.echo(f"Removed the stored {SERVICE_LABELS[service.value]} sign-in from this computer.")
    session.echo(REVOKE_HINTS[service.value])


def _forget(store: TokenStore, service: Service) -> None:
    for key in STORE_KEYS[service]:
        store.clear(key)


def main() -> None:
    """Entry point for ``scripts/connect_accounts.py``."""
    connect_app()


# Design notes:
# - Sign-in runs on the main thread (acquire_blocking), not through the
#   async access_token(): Ctrl+C then interrupts the browser wait or the
#   device-code poll directly instead of leaving a worker thread polling.
# - The access token is added to the redaction list the moment it exists,
#   so even an error body that echoes it back prints "[hidden]".
# - Discord "connected" is the result of the last successful check, kept in
#   the token store: there is no token to store, but status should still
#   say whether the setup was proven to work.
# - The deps travel as Typer's context object so the same commands serve
#   the script, the `jarvis connect` sub-command and the tests.
