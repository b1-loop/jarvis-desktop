"""Google OAuth for Gmail and Calendar, shared through one token source.

The installed-app flow (browser consent, local redirect) is blocking and
interactive, so it runs in a worker thread; the resulting authorized-user
JSON is kept in the :class:`TokenStore` and refreshed silently afterwards.
A non-interactive source (the windowless app, voice mode) never starts the
flow: when consent is needed it raises
:class:`~jarvis.integrations.base.ConsentRequiredError` naming the connect
command, because a browser opened by a hidden process is a sign-in nobody
asked for. The three places the google libraries are touched (load,
refresh, run the flow) are injectable, which is how the tests cover every
branch without a browser or a network.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from google.auth.exceptions import GoogleAuthError, TransportError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from jarvis.integrations.base import AuthError, NotConfiguredError, ProviderError
from jarvis.integrations.consent import ConnectionLog, consent_required
from jarvis.integrations.tokens import ThreadedTokenSource, TokenStore
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import Settings

log = get_logger(__name__)

PROVIDER = "google"
STORE_KEY = "google"


class GoogleCredentials(Protocol):
    """The slice of ``google.oauth2.credentials.Credentials`` Jarvis relies on."""

    @property
    def token(self) -> str | None:
        """Current access token."""

    @property
    def valid(self) -> bool:
        """True if a token is present and not expired."""

    @property
    def refresh_token(self) -> str | None:
        """Long-lived token for silent refresh."""

    def refresh(self, request: Any) -> None:
        """Obtain a fresh access token."""

    def to_json(self) -> str:
        """Authorized-user JSON suitable for ``from_authorized_user_info``."""


CredentialsLoader = Callable[[dict[str, Any], Sequence[str]], GoogleCredentials]
CredentialsRefresher = Callable[[GoogleCredentials], None]
FlowRunner = Callable[[Path, Sequence[str]], GoogleCredentials]


def load_credentials(info: dict[str, Any], scopes: Sequence[str]) -> GoogleCredentials:
    """Rebuild credentials from stored authorized-user JSON."""
    # google-auth ships py.typed but leaves this classmethod unannotated.
    credentials = Credentials.from_authorized_user_info(info, list(scopes))  # type: ignore[no-untyped-call]
    return cast(GoogleCredentials, credentials)


def refresh_credentials(credentials: GoogleCredentials) -> None:
    """Refresh over the network.

    Raises:
        AuthError: If Google rejected the refresh token (revoked, expired).
        ProviderError: If Google could not be reached; the token may still be
            good, so it must not be thrown away.
    """
    try:
        credentials.refresh(Request())
    except TransportError as error:
        raise ProviderError(PROVIDER, f"token refresh could not reach Google: {error}") from error
    except GoogleAuthError as error:
        raise AuthError(PROVIDER, f"token refresh rejected: {error}") from error


def run_installed_app_flow(client_secrets_file: Path, scopes: Sequence[str]) -> GoogleCredentials:
    """Open the browser for consent and wait on a local redirect server.

    Raises:
        AuthError: If the secrets file is unusable or consent fails.
    """
    try:
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets_file), list(scopes))
        return cast(GoogleCredentials, flow.run_local_server(port=0))
    except (OSError, ValueError, GoogleAuthError) as error:
        raise AuthError(PROVIDER, f"OAuth consent failed: {error}") from error


class GoogleAuth(ThreadedTokenSource):
    """Token source backed by the installed-app flow and a token store.

    Args:
        client_secrets_file: The OAuth client JSON downloaded from Google Cloud.
        scopes: Every scope the enabled Google providers need, requested in
            one consent so the user is asked once.
        token_store: Where the authorized-user JSON is kept between runs.
        load: Seam for ``Credentials.from_authorized_user_info``.
        refresh: Seam for the network refresh.
        run_flow: Seam for the interactive consent flow.
        interactive: Whether consent may be asked for here. ``False`` raises
            :class:`~jarvis.integrations.base.ConsentRequiredError` instead.
    """

    def __init__(
        self,
        client_secrets_file: Path,
        scopes: Sequence[str],
        token_store: TokenStore,
        *,
        load: CredentialsLoader = load_credentials,
        refresh: CredentialsRefresher = refresh_credentials,
        run_flow: FlowRunner = run_installed_app_flow,
        interactive: bool = True,
    ) -> None:
        super().__init__()
        self._client_secrets_file = client_secrets_file
        self._scopes = tuple(scopes)
        self._store = token_store
        self._load = load
        self._refresh = refresh
        self._run_flow = run_flow
        self._interactive = interactive
        self._log = ConnectionLog(token_store)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        scopes: Sequence[str],
        token_store: TokenStore,
        *,
        interactive: bool = True,
    ) -> GoogleAuth:
        """Build from settings; see the class for ``interactive``.

        Raises:
            NotConfiguredError: If ``GOOGLE_CLIENT_SECRETS_FILE`` is unset.
        """
        if settings.google_client_secrets_file is None:
            raise NotConfiguredError(PROVIDER, ["GOOGLE_CLIENT_SECRETS_FILE"])
        return cls(
            settings.google_client_secrets_file, scopes, token_store, interactive=interactive
        )

    @property
    def scopes(self) -> tuple[str, ...]:
        """Scopes that will be requested at consent time."""
        return self._scopes

    @property
    def interactive(self) -> bool:
        """Whether this source may open the browser for consent."""
        return self._interactive

    def _acquire_blocking(self) -> str:
        credentials = self._stored_credentials()
        had_stored = credentials is not None
        if credentials is not None and not credentials.valid:
            credentials = self._refreshed_or_none(credentials)
        if credentials is None:
            credentials = self._consent(expired=had_stored)
        self._store.save(STORE_KEY, json.loads(credentials.to_json()))
        if not credentials.token:
            raise AuthError(PROVIDER, "consent completed without an access token")
        self._log.clear(PROVIDER)
        return credentials.token

    def _consent(self, *, expired: bool) -> GoogleCredentials:
        """Run the flow, or say how to, depending on the mode."""
        log.info(
            "integrations.google.consent_required",
            scopes=self._scopes,
            interactive=self._interactive,
        )
        if not self._interactive:
            reason = "stored token can no longer be refreshed" if expired else "no stored token"
            error = consent_required(PROVIDER, reason, expired=expired)
            self._log.record(PROVIDER, error.remedy)
            raise error
        return self._run_flow(self._client_secrets_file, self._scopes)

    def _stored_credentials(self) -> GoogleCredentials | None:
        info = self._store.load(STORE_KEY)
        if info is None:
            return None
        granted = set(info.get("scopes", []))
        if not granted.issuperset(self._scopes):
            log.info(
                "integrations.google.scopes_changed", missing=sorted(set(self._scopes) - granted)
            )
            return None
        return self._load(info, self._scopes)

    def _refreshed_or_none(self, credentials: GoogleCredentials) -> GoogleCredentials | None:
        if not credentials.refresh_token:
            return None
        try:
            self._refresh(credentials)
        except AuthError as error:
            # A revoked refresh token is recoverable by consenting again, so
            # forget it and fall through to consent instead of failing.
            log.warning("integrations.google.refresh_failed", error=str(error))
            self._store.clear(STORE_KEY)
            return None
        return credentials


# Design notes:
# - One GoogleAuth for Gmail and Calendar: a single consent screen with the
#   union of both scope sets. The alternative, two tokens with narrower scopes
#   each, doubles the consent prompts for no privacy gain since the same
#   Google account grants both.
# - A stored token whose granted scopes no longer cover the required set is
#   treated as absent. Reusing it would produce confusing 403s later.
# - Non-interactive mode only changes what happens when consent is needed;
#   stored tokens and silent refresh behave the same in both modes, so the
#   desktop app runs on what connect_accounts.py stored. A network failure
#   during refresh is a ProviderError and keeps the token: being offline for
#   a minute must not cost the user a sign-in.
# - The google libraries are imported at module level; this module is only
#   imported by the factory when GOOGLE_CLIENT_SECRETS_FILE is set, so the
#   stub path never needs the optional extra.
