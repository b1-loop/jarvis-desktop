import json
from collections.abc import Callable
from typing import Any

import httpx
import numpy as np
import pytest

from jarvis.llm.embeddings import EmbeddingError, EmbeddingModelUnavailableError, normalised
from jarvis.llm.ollama_embeddings import EMBED_PATH, KEEP_ALIVE, OllamaEmbedder


class FakeServer:
    """``httpx.MockTransport`` handler with scripted replies; records request bodies."""

    def __init__(self, *replies: httpx.Response | Exception) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.paths: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        self.requests.append(json.loads(request.content))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def make_embedder(server: FakeServer, model: str = "bge-m3") -> OllamaEmbedder:
    http = httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(server))
    return OllamaEmbedder(http, model=model)


def vectors(*rows: list[float]) -> httpx.Response:
    return httpx.Response(200, json={"model": "bge-m3", "embeddings": list(rows)})


async def test_documents_are_embedded_in_one_request_and_normalised() -> None:
    server = FakeServer(vectors([3.0, 4.0], [0.0, 2.0]))
    embedder = make_embedder(server)

    result = await embedder.embed_documents(["a", "b"])

    assert server.paths == [EMBED_PATH]
    assert server.requests[0] == {"model": "bge-m3", "input": ["a", "b"], "keep_alive": KEEP_ALIVE}
    np.testing.assert_allclose(result[0], [0.6, 0.8])
    np.testing.assert_allclose(result[1], [0.0, 1.0])
    assert embedder.model == "bge-m3"
    await embedder.aclose()


async def test_no_documents_need_no_request() -> None:
    server = FakeServer()

    assert await make_embedder(server).embed_documents([]) == []
    assert server.requests == []


async def test_a_query_is_a_batch_of_one() -> None:
    server = FakeServer(vectors([1.0, 0.0]))

    vector = await make_embedder(server).embed_query("hur mycket batteri har jag?")

    assert server.requests[0]["input"] == ["hur mycket batteri har jag?"]
    np.testing.assert_allclose(vector, [1.0, 0.0])


async def test_models_trained_with_prompts_get_them() -> None:
    server = FakeServer(vectors([1.0]), vectors([1.0]))
    embedder = make_embedder(server, model="embeddinggemma:latest")

    await embedder.embed_query("vad är klockan")
    await embedder.embed_documents(["current time: Get the time."])

    assert server.requests[0]["input"] == ["task: search result | query: vad är klockan"]
    assert server.requests[1]["input"] == ["title: none | text: current time: Get the time."]


@pytest.mark.parametrize("status", [400, 404, 501])
async def test_a_missing_or_unsuitable_model_is_permanent_and_names_the_fix(status: int) -> None:
    server = FakeServer(httpx.Response(status, json={"error": "model 'bge-m3' not found"}))

    with pytest.raises(EmbeddingModelUnavailableError, match="ollama pull bge-m3"):
        await make_embedder(server).embed_query("x")


async def test_server_errors_are_transient() -> None:
    server = FakeServer(httpx.Response(503, text="busy"))

    with pytest.raises(EmbeddingError) as raised:
        await make_embedder(server).embed_query("x")

    assert not isinstance(raised.value, EmbeddingModelUnavailableError)
    assert "503" in str(raised.value)


async def test_an_unreachable_server_is_transient() -> None:
    server = FakeServer(httpx.ConnectError("refused"))

    with pytest.raises(EmbeddingError, match="could not reach Ollama"):
        await make_embedder(server).embed_query("x")


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"embeddings": []}),
        httpx.Response(200, json=["no", "object"]),
    ],
)
async def test_unusable_replies_are_errors(reply: httpx.Response) -> None:
    with pytest.raises(EmbeddingError):
        await make_embedder(FakeServer(reply)).embed_query("x")


def test_a_zero_vector_stays_zero() -> None:
    np.testing.assert_array_equal(normalised([0.0, 0.0]), [0.0, 0.0])


async def test_from_settings_points_at_the_configured_server(
    make_settings: Callable[..., Any],
) -> None:
    embedder = OllamaEmbedder.from_settings(
        make_settings(ollama_base_url="http://box:11434"), "bge-m3"
    )

    assert embedder.model == "bge-m3"
    await embedder.aclose()
