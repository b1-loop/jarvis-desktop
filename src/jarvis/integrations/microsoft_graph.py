"""Microsoft identity (msal) and Graph helpers shared by Outlook and Teams.

A personal Jarvis has no web server to receive an OAuth redirect, so the
device-code flow is used: msal hands back a URL and a short code, the user
signs in on any device, and the blocking poll runs in a worker thread. The
msal token cache (which holds the refresh token) is serialised into the
:class:`TokenStore` so the user signs in once. A non-interactive source (the
windowless app, voice mode) never starts the device flow: a code written to
a log file is a sign-in nobody completes, so it raises
:class:`~jarvis.integrations.base.ConsentRequiredError` naming the connect
command instead.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

import msal

from jarvis.integrations.base import AuthError, NotConfiguredError
from jarvis.integrations.consent import ConnectionLog, consent_required
from jarvis.integrations.tokens import ThreadedTokenSource, TokenStore
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.config import Settings

log = get_logger(__name__)

PROVIDER = "microsoft"
STORE_KEY = "microsoft"
GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
AUTHORITY_TEMPLATE = "https://login.microsoftonline.com/{tenant}"
DEFAULT_VERIFICATION_URI = "https://microsoft.com/devicelogin"

_FRACTION = re.compile(r"\.(\d{6})\d*")


class MsalApp(Protocol):
    """The slice of ``msal.PublicClientApplication`` Jarvis relies on."""

    def get_accounts(self) -> list[dict[str, Any]]:
        """Accounts present in the token cache."""

    def acquire_token_silent(
        self, scopes: list[str], account: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Token from cache or refresh token; ``None`` if interaction is needed."""

    def initiate_device_flow(self, scopes: list[str]) -> dict[str, Any]:
        """Start the device-code flow; the dict carries ``user_code`` and ``message``."""

    def acquire_token_by_device_flow(self, flow: dict[str, Any]) -> dict[str, Any]:
        """Poll until the user completes sign-in."""


class TokenCache(Protocol):
    """The slice of ``msal.SerializableTokenCache`` Jarvis relies on."""

    @property
    def has_state_changed(self) -> bool:
        """True after msal wrote to the cache."""

    def serialize(self) -> str:
        """Cache contents as JSON text."""

    def deserialize(self, state: str | None) -> None:
        """Replace the cache contents."""


@dataclass(frozen=True, slots=True)
class DeviceCode:
    """What the user needs to finish a device-code sign-in.

    Only the parts meant for the user: msal's ``device_code`` (the secret
    the poll uses) never leaves the flow dict.

    Attributes:
        user_code: The short code to type on the sign-in page.
        verification_uri: Where to type it.
        message: msal's own one-line instructions.
        expires_in: Seconds until the code stops working, when known.
    """

    user_code: str
    verification_uri: str
    message: str
    expires_in: int | None = None

    @classmethod
    def from_flow(cls, flow: dict[str, Any]) -> DeviceCode:
        """Pick the user-facing fields out of msal's flow dict."""
        expires_in = flow.get("expires_in")
        return cls(
            user_code=str(flow["user_code"]),
            verification_uri=str(flow.get("verification_uri") or DEFAULT_VERIFICATION_URI),
            message=str(flow.get("message", "")),
            expires_in=int(expires_in) if expires_in is not None else None,
        )


AppFactory = Callable[[TokenCache], MsalApp]
CacheFactory = Callable[[], TokenCache]
Prompt = Callable[[DeviceCode], None]


def new_token_cache() -> TokenCache:
    """A fresh, empty msal cache."""
    return cast(TokenCache, msal.SerializableTokenCache())


def build_public_client(client_id: str, tenant_id: str, cache: TokenCache) -> MsalApp:
    """Construct the real msal app.

    Called from the worker thread on first use because msal performs tenant
    discovery over the network in the constructor.
    """
    app = msal.PublicClientApplication(
        client_id, authority=AUTHORITY_TEMPLATE.format(tenant=tenant_id), token_cache=cache
    )
    return cast(MsalApp, app)


def log_device_code_prompt(code: DeviceCode) -> None:
    """Default prompt for an interactive source without a terminal renderer."""
    log.warning("integrations.microsoft.sign_in_required", instructions=code.message)


class MicrosoftAuth(ThreadedTokenSource):
    """Token source over msal's device-code flow and a persisted cache.

    Args:
        app_factory: Builds the msal client around a cache, on first use.
        scopes: Every Graph scope the enabled providers need.
        token_store: Where the serialised cache is kept.
        cache_factory: Builds the (empty) msal cache the app is created with.
        prompt: How to show the user the device code (interactive only).
        interactive: Whether a device-code sign-in may start here. ``False``
            raises :class:`~jarvis.integrations.base.ConsentRequiredError`.
    """

    def __init__(
        self,
        app_factory: AppFactory,
        scopes: Sequence[str],
        token_store: TokenStore,
        *,
        cache_factory: CacheFactory = new_token_cache,
        prompt: Prompt = log_device_code_prompt,
        interactive: bool = True,
    ) -> None:
        super().__init__()
        self._app_factory = app_factory
        self._cache_factory = cache_factory
        self._scopes = list(scopes)
        self._store = token_store
        self._prompt = prompt
        self._interactive = interactive
        self._log = ConnectionLog(token_store)
        self._app: MsalApp | None = None
        self._cache: TokenCache | None = None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        scopes: Sequence[str],
        token_store: TokenStore,
        *,
        prompt: Prompt = log_device_code_prompt,
        interactive: bool = True,
    ) -> MicrosoftAuth:
        """Build a lazily constructed real msal client from settings.

        Raises:
            NotConfiguredError: If ``MS_CLIENT_ID`` is unset.
        """
        if settings.ms_client_id is None:
            raise NotConfiguredError(PROVIDER, ["MS_CLIENT_ID"])
        factory = functools.partial(
            build_public_client, settings.ms_client_id, settings.ms_tenant_id
        )
        return cls(factory, scopes, token_store, prompt=prompt, interactive=interactive)

    @property
    def scopes(self) -> tuple[str, ...]:
        """Scopes requested at sign-in."""
        return tuple(self._scopes)

    @property
    def interactive(self) -> bool:
        """Whether this source may start a device-code sign-in."""
        return self._interactive

    def signed_in_username(self) -> str | None:
        """The cached account's sign-in name (blocking; call after a token was issued)."""
        app, _cache = self._ensure_app()
        accounts = app.get_accounts()
        return str(accounts[0].get("username")) if accounts else None

    def _acquire_blocking(self) -> str:
        app, cache = self._ensure_app()
        self._load_cache(cache)
        accounts = app.get_accounts()
        result = app.acquire_token_silent(self._scopes, accounts[0]) if accounts else None
        if not result or not result.get("access_token"):
            result = self._consent(app, result, has_account=bool(accounts))
        self._save_cache(cache)
        token = result.get("access_token")
        if not token:
            raise AuthError(PROVIDER, _describe_failure(result))
        self._log.clear(PROVIDER)
        return str(token)

    def _consent(
        self, app: MsalApp, silent: dict[str, Any] | None, *, has_account: bool
    ) -> dict[str, Any]:
        """Start the device flow, or say how to, depending on the mode."""
        log.info(
            "integrations.microsoft.consent_required",
            scopes=self._scopes,
            interactive=self._interactive,
        )
        if self._interactive:
            return self._device_flow(app)
        reason = _describe_failure(silent) if silent else "no cached account"
        error = consent_required(PROVIDER, reason, expired=has_account)
        self._log.record(PROVIDER, error.remedy)
        raise error

    def _ensure_app(self) -> tuple[MsalApp, TokenCache]:
        if self._app is None or self._cache is None:
            self._cache = self._cache_factory()
            self._app = self._app_factory(self._cache)
        return self._app, self._cache

    def _device_flow(self, app: MsalApp) -> dict[str, Any]:
        flow = app.initiate_device_flow(self._scopes)
        if "user_code" not in flow:
            raise AuthError(PROVIDER, f"could not start device flow: {_describe_failure(flow)}")
        self._prompt(DeviceCode.from_flow(flow))
        return app.acquire_token_by_device_flow(flow)

    def _load_cache(self, cache: TokenCache) -> None:
        stored = self._store.load(STORE_KEY)
        if stored is not None:
            cache.deserialize(stored.get("cache"))

    def _save_cache(self, cache: TokenCache) -> None:
        if cache.has_state_changed:
            self._store.save(STORE_KEY, {"cache": cache.serialize()})


def _describe_failure(result: dict[str, Any]) -> str:
    return str(result.get("error_description") or result.get("error") or "unknown error")


def parse_graph_datetime(value: str, *, assume_utc: bool = True) -> datetime:
    """Parse Graph's ``2026-10-21T10:00:00.0000000`` (seven fraction digits).

    Graph returns seven fractional digits and, for calendar values, no
    offset; the offset is whatever ``Prefer: outlook.timezone`` asked for.
    """
    trimmed = _FRACTION.sub(r".\1", value.replace("Z", "+00:00"))
    parsed = datetime.fromisoformat(trimmed)
    if parsed.tzinfo is None and assume_utc:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


# Design notes:
# - Device-code over authorization-code: no redirect listener to manage and it
#   works when Jarvis runs on a machine without a browser session (remote
#   desktop, headless test box).
# - The msal app is built on first use, in the worker thread: its constructor
#   does tenant discovery over HTTP, which must not happen at startup or in a
#   unit test that merely wires providers together.
# - The whole msal cache is persisted as one opaque string. Picking fields out
#   of it would couple Jarvis to msal's cache schema for no benefit.
# - Scopes are the union across Outlook and (when enabled) Teams so the user
#   consents once; see ADR-0009 for why that is still least privilege.
# - The prompt receives a DeviceCode rather than msal's message string so a
#   terminal can show the code and URL prominently (and copy the code)
#   without parsing English text; device_code itself is never passed on.
# - A silent result without a token (msal's "interaction_required", a
#   revoked refresh token) means "consent needed" in both modes: signing in
#   again is the only thing the user can do about it.
