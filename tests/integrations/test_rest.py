import httpx
import pytest

from jarvis.integrations.base import AuthError, ProviderError, RateLimitedError
from jarvis.integrations.rest import RestSession
from jarvis.integrations.tokens import StaticTokenSource
from tests.integrations.conftest import FAKE_TOKEN, Router


def session(router: Router, token_source: StaticTokenSource, **kwargs: str) -> RestSession:
    return RestSession(
        router.client(),
        provider="demo",
        base_url="https://api.example.test/v1/",
        token_source=token_source,
        **kwargs,
    )


async def test_request_adds_bearer_token_and_joins_base_url(
    router: Router, token_source: StaticTokenSource
) -> None:
    router.add("GET", "/v1/things", {"ok": True})
    rest = session(router, token_source)

    data = await rest.request("GET", "/things", params={"q": "x"}, headers={"Prefer": "y"})

    assert data == {"ok": True}
    sent = router.last()
    assert sent.headers["Authorization"] == f"Bearer {FAKE_TOKEN}"
    assert sent.headers["Prefer"] == "y"
    assert str(sent.url) == "https://api.example.test/v1/things?q=x"
    status = rest.status()
    assert status.configured and status.authenticated and status.last_error is None


async def test_absolute_urls_and_custom_auth_scheme(
    router: Router, token_source: StaticTokenSource
) -> None:
    router.add("POST", "/elsewhere", {"id": 1})
    rest = session(router, token_source, auth_scheme="Bot")

    await rest.request("POST", "https://other.example.test/elsewhere", json={"a": 1})

    assert router.last().headers["Authorization"] == f"Bot {FAKE_TOKEN}"
    assert router.sent_json() == {"a": 1}


async def test_empty_and_no_content_responses_decode_to_none(
    router: Router, token_source: StaticTokenSource
) -> None:
    router.add("POST", "/v1/send", status=202)
    router.add("GET", "/v1/idle", status=204)
    rest = session(router, token_source)

    assert await rest.request("POST", "/send") is None
    assert await rest.request("GET", "/idle") is None


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, AuthError), (403, AuthError), (429, RateLimitedError), (500, ProviderError)],
)
async def test_statuses_map_to_the_error_hierarchy(
    router: Router, token_source: StaticTokenSource, status: int, expected: type[Exception]
) -> None:
    router.add(
        "GET", "/v1/x", status=status, text="insufficient scope", headers={"Retry-After": "7"}
    )
    rest = session(router, token_source)

    with pytest.raises(expected) as raised:
        await rest.request("GET", "/x")

    assert f"HTTP {status}" in str(raised.value)
    assert rest.status().last_error == str(raised.value)
    assert not rest.status().authenticated
    if isinstance(raised.value, RateLimitedError):
        assert raised.value.retry_after == 7.0
    if isinstance(raised.value, ProviderError):
        assert raised.value.status_code == status


async def test_bad_retry_after_header_is_ignored(
    router: Router, token_source: StaticTokenSource
) -> None:
    router.add("GET", "/v1/x", status=429, headers={"Retry-After": "soon"})

    with pytest.raises(RateLimitedError) as raised:
        await session(router, token_source).request("GET", "/x")

    assert raised.value.retry_after is None


async def test_transport_errors_become_provider_errors(token_source: StaticTokenSource) -> None:
    def explode(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(explode))
    rest = RestSession(client, provider="demo", base_url="https://x", token_source=token_source)

    with pytest.raises(ProviderError, match="request failed"):
        await rest.request("GET", "/x")


async def test_error_text_never_contains_the_token(
    router: Router, token_source: StaticTokenSource
) -> None:
    router.add("GET", "/v1/x", status=500, text="server exploded")

    with pytest.raises(ProviderError) as raised:
        await session(router, token_source).request("GET", "/x")

    assert FAKE_TOKEN not in str(raised.value)


async def test_success_after_failure_clears_last_error(
    router: Router, token_source: StaticTokenSource
) -> None:
    router.add("GET", "/v1/bad", status=500)
    router.add("GET", "/v1/good", {"ok": 1})
    rest = session(router, token_source)

    with pytest.raises(ProviderError):
        await rest.request("GET", "/bad")
    await rest.request("GET", "/good")

    assert rest.status().last_error is None
