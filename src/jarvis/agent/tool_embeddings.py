"""Multilingual semantic tool ranking: cosine similarity in embedding space.

:class:`EmbeddingToolRanker` is the :class:`~jarvis.agent.tool_selection.SemanticScorer`
behind the hybrid tool selection (ADR-0020, update). Each tool's name,
description and keywords are embedded once, at startup, and kept in a
cache file under ``data_dir`` so later starts skip the work; each request
is embedded per turn, within a latency budget. A request in Swedish then
finds the English description of the tool it needs, which word overlap
cannot do.

The ranker never makes a turn wait for long and never fails one: if the
model is missing, Ollama is down or the embedding is late, it returns
``None`` and the selector ranks by words alone.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from jarvis.llm.embeddings import (
    EmbeddingError,
    EmbeddingModelUnavailableError,
    TextEmbedder,
    Vector,
)
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from jarvis.agent.registry import AnyTool

log = get_logger(__name__)

DEFAULT_BUDGET_SECONDS: Final = 0.15
"""Longest a turn waits for its request embedding (warm ``bge-m3``: about 25 ms)."""

RETRY_AFTER_SECONDS: Final = 60.0
"""Pause after a transient failure before the next attempt, so a down server costs nothing."""

_REQUEST_MEMO_SIZE: Final = 8
"""Request vectors remembered; a turn re-selects tools every round with the same words."""

CACHE_FORMAT: Final = 1


def tool_document(tool: AnyTool) -> str:
    """The text a tool is embedded as: spoken name, description and keywords."""
    keywords = f" Keywords: {', '.join(tool.keywords)}." if tool.keywords else ""
    return f"{tool.spoken_name}: {tool.description}{keywords}"


def document_key(model: str, document: str) -> str:
    """Cache key of one embedded document; changes with the model or the text."""
    return hashlib.sha256(f"{model}\n{document}".encode()).hexdigest()


class ToolVectorCache:
    """Tool vectors on disk, as JSON keyed by :func:`document_key`.

    Args:
        path: The cache file; its directory is created on first save.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[str, Vector]:
        """Every stored vector; an absent, unreadable or foreign file is an empty cache."""
        try:
            raw: Any = json.loads(self.path.read_text("utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as error:
            log.warning("tool_selection.cache_unreadable", path=str(self.path), error=str(error))
            return {}
        if not isinstance(raw, dict) or raw.get("format") != CACHE_FORMAT:
            return {}
        stored = raw.get("vectors")
        if not isinstance(stored, dict):
            return {}
        return {key: np.asarray(values, dtype=np.float32) for key, values in stored.items()}

    def save(self, vectors: Mapping[str, Vector]) -> None:
        """Replace the file atomically with exactly ``vectors``."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CACHE_FORMAT,
            "vectors": {key: vector.tolist() for key, vector in vectors.items()},
        }
        partial = self.path.with_suffix(".tmp")
        partial.write_text(json.dumps(payload), "utf-8")
        partial.replace(self.path)


@dataclass(slots=True)
class _Health:
    """Whether the embedding service may be asked, with log-once bookkeeping."""

    clock: Callable[[], float]
    retry_after: float
    disabled: bool = False
    retry_at: float = 0.0
    reported: bool = False
    timeouts: int = 0
    seen: set[str] = field(default_factory=set)

    def usable(self) -> bool:
        return not self.disabled and self.clock() >= self.retry_at

    def failed(self, error: EmbeddingError, model: str) -> None:
        """Back off (or give up for good) and say so once, not every turn."""
        if isinstance(error, EmbeddingModelUnavailableError):
            self.disabled = True
        else:
            self.retry_at = self.clock() + self.retry_after
        if not self.reported:
            self.reported = True
            log.warning(
                "tool_selection.semantic_unavailable",
                model=model,
                error=str(error),
                fallback="keywords",
                permanent=self.disabled,
            )

    def recovered(self, model: str) -> None:
        if self.reported:
            self.reported = False
            log.info("tool_selection.semantic_recovered", model=model)

    def late(self, budget: float) -> None:
        """A request embedding missed its budget: info the first time, debug after."""
        self.timeouts += 1
        level = log.info if self.timeouts == 1 else log.debug
        level(
            "tool_selection.semantic_timeout", budget_ms=round(budget * 1000), count=self.timeouts
        )


class EmbeddingToolRanker:
    """Cosine similarity between a request and each tool, via a ``TextEmbedder``.

    Args:
        embedder: The embedding model client.
        cache: Where tool vectors persist between runs; ``None`` keeps them in memory.
        budget_seconds: Longest :meth:`similarities` waits for a request vector.
        retry_after_seconds: Pause after a transient failure.
        clock: Monotonic seconds, injectable for tests.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        *,
        cache: ToolVectorCache | None = None,
        budget_seconds: float = DEFAULT_BUDGET_SECONDS,
        retry_after_seconds: float = RETRY_AFTER_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.embedder = embedder
        self.cache = cache
        self.budget_seconds = budget_seconds
        self._health = _Health(clock=clock, retry_after=retry_after_seconds)
        self._vectors: dict[str, Vector] | None = None
        self._preparing: asyncio.Task[bool] | None = None
        self._requests: OrderedDict[str, asyncio.Task[Vector]] = OrderedDict()

    def warm_up(self, tools: Sequence[AnyTool]) -> None:
        """Start embedding ``tools`` (and loading the model) in the background."""
        if self._preparing is None or self._preparing.done():
            self._preparing = asyncio.create_task(self.prepare(tools), name="tool-embeddings")

    async def prepare(self, tools: Sequence[AnyTool]) -> bool:
        """Make sure every tool has a vector: from the cache, else from the model.

        Returns:
            True when every tool can be ranked semantically.
        """
        if self._vectors is None:
            self._vectors = await self._load_cache()
        documents = {self._key(tool): tool_document(tool) for tool in tools}
        missing = [key for key in documents if key not in self._vectors]
        if missing:
            try:
                fresh = await self.embedder.embed_documents([documents[key] for key in missing])
            except EmbeddingError as error:
                self._health.failed(error, self.embedder.model)
                return False
            self._vectors.update(zip(missing, fresh, strict=True))
            log.info("tool_selection.tools_embedded", model=self.embedder.model, count=len(missing))
            self._health.recovered(self.embedder.model)
        if missing or self._vectors.keys() != documents.keys():
            self._vectors = {key: self._vectors[key] for key in documents}
            await self._save_cache(self._vectors)
        return True

    async def similarities(
        self, request: str, tools: Sequence[AnyTool]
    ) -> Mapping[str, float] | None:
        """Cosine similarity per tool name, or ``None`` to rank by words this turn."""
        if not self._health.usable():
            return None
        tool_vectors = self._tool_vectors(tools)
        if tool_vectors is None:
            self.warm_up(tools)
            return None
        query = await self._request_vector(request)
        if query is None:
            return None
        return {name: float(vector @ query) for name, vector in tool_vectors.items()}

    async def aclose(self) -> None:
        """Cancel background work (tool warm-up, late request embeddings)."""
        pending = [task for task in (self._preparing, *self._requests.values()) if task]
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(asyncio.CancelledError, EmbeddingError):
                await task

    def _tool_vectors(self, tools: Sequence[AnyTool]) -> dict[str, Vector] | None:
        if self._vectors is None:
            return None
        found = {tool.name: self._vectors.get(self._key(tool)) for tool in tools}
        if any(vector is None for vector in found.values()):
            return None
        return {name: vector for name, vector in found.items() if vector is not None}

    async def _request_vector(self, request: str) -> Vector | None:
        """The request's vector if it is ready within the budget.

        A late embedding is not cancelled: it finishes in the background,
        loads the model if that was the delay, and serves the next round of
        the same turn from the memo.
        """
        task = self._request_task(request)
        done, _ = await asyncio.wait({task}, timeout=self.budget_seconds)
        if not done:
            self._health.late(self.budget_seconds)
            return None
        try:
            vector = task.result()
        except EmbeddingError as error:
            self._requests.pop(request, None)
            self._health.failed(error, self.embedder.model)
            return None
        self._health.recovered(self.embedder.model)
        return vector

    def _request_task(self, request: str) -> asyncio.Task[Vector]:
        task = self._requests.get(request)
        if task is None:
            task = asyncio.create_task(self.embedder.embed_query(request), name="request-embedding")
            task.add_done_callback(_retrieve_exception)
            self._requests[request] = task
            if len(self._requests) > _REQUEST_MEMO_SIZE:
                _, oldest = self._requests.popitem(last=False)
                if not oldest.done():
                    oldest.cancel()
        self._requests.move_to_end(request)
        return task

    def _key(self, tool: AnyTool) -> str:
        return document_key(self.embedder.model, tool_document(tool))

    async def _load_cache(self) -> dict[str, Vector]:
        if self.cache is None:
            return {}
        return await asyncio.to_thread(self.cache.load)

    async def _save_cache(self, vectors: Mapping[str, Vector]) -> None:
        if self.cache is None:
            return
        try:
            await asyncio.to_thread(self.cache.save, vectors)
        except OSError as error:
            log.warning(
                "tool_selection.cache_unwritable", path=str(self.cache.path), error=str(error)
            )


def _retrieve_exception(task: asyncio.Task[Vector]) -> None:
    """Mark a late task's failure as seen; the next lookup reports it, or nobody needs it."""
    if not task.cancelled():
        task.exception()


# Design notes:
# - The budget is enforced with ``asyncio.wait`` rather than a cancelling
#   timeout: the usual reason for a late vector is Ollama loading the model,
#   and cancelling would abort that load and make every turn late. The
#   result is memoised, so later rounds of the same turn (which re-select
#   tools) get the semantic ranking once it exists.
# - Tool vectors are keyed by a hash of model and text, so editing a
#   description, adding a tool or switching the model re-embeds exactly
#   what changed; the saved file holds only the current tools' vectors.
# - A cold start never blocks: until the background warm-up has embedded
#   the tools, every turn ranks by words, exactly as before this ranker.
# - Failures are logged once per outage (warning) and once on recovery
#   (info); a missing model disables the ranker for the session because
#   pulling it needs the user, and the warning says which command to run.
# - JSON, not pickle or ``np.save``: the cache is data from disk, and JSON
#   cannot execute anything when loaded. About 0.6 MB for 62 tools.
