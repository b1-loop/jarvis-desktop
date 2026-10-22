"""Shared helpers for provider tests.

Every REST provider is exercised through ``httpx.MockTransport``; ``Router``
maps ``(method, path)`` to canned responses and records what was sent so
tests assert on request shaping (query, headers, body) as well as parsing.
"""

from __future__ import annotations

import json as jsonlib
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from jarvis.integrations.tokens import StaticTokenSource

FAKE_TOKEN = "fake-token-not-real"

Responder = Callable[[httpx.Request], httpx.Response]


class Router:
    """Canned responses by ``(METHOD, path)`` plus a record of every request."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._routes: dict[tuple[str, str], Responder] = {}

    def add(
        self,
        method: str,
        path: str,
        json: Any = None,
        *,
        status: int = 200,
        text: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        def respond(_request: httpx.Request) -> httpx.Response:
            if text is not None:
                return httpx.Response(status, text=text, headers=headers)
            if json is None:
                return httpx.Response(status, headers=headers)
            return httpx.Response(status, json=json, headers=headers)

        self._routes[(method.upper(), path)] = respond

    def add_dynamic(self, method: str, path: str, respond: Responder) -> None:
        self._routes[(method.upper(), path)] = respond

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        responder = self._routes.get((request.method, request.url.path))
        if responder is None:
            return httpx.Response(599, text=f"unrouted {request.method} {request.url}")
        return responder(request)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handle))

    def last(self) -> httpx.Request:
        return self.requests[-1]

    def sent_json(self, index: int = -1) -> Any:
        return jsonlib.loads(self.requests[index].content)


@pytest.fixture
def router() -> Router:
    return Router()


@pytest.fixture
def token_source() -> StaticTokenSource:
    return StaticTokenSource(SecretStr(FAKE_TOKEN))
