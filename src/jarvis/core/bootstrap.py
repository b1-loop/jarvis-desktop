"""Composition root: builds every component from ``Settings`` and wires them together.

This is the one module (with ``__main__``, which only calls it) that knows
the concrete classes: which LLM client, which vector store, which OS
backend, which integration providers. Each ``build_*`` function makes one
choice, logs it, registers the component's cleanup on an
``AsyncExitStack`` and returns the interface. Everything downstream
receives its dependencies by injection.

Assembly order follows the dependencies: settings -> event bus -> memory
-> LLM -> tool registry -> platform -> integrations -> agent ->
orchestrator.
"""

from __future__ import annotations

import importlib.util
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

import httpx

from jarvis.agent.confirmation import ConfirmationPolicy
from jarvis.agent.loop import Agent, AgentLimits
from jarvis.agent.persona import Persona
from jarvis.agent.registry import ToolRegistry
from jarvis.agent.tool import ConfirmationAsker, ToolContext
from jarvis.config import Settings
from jarvis.core.events import EventBus
from jarvis.core.orchestrator import (
    InputSource,
    Orchestrator,
    OutputSink,
    TerminalConfirmation,
    TerminalInput,
    TerminalOutput,
)
from jarvis.core.state_machine import StateMachine
from jarvis.integrations.services import IntegrationServices, build_integration_services
from jarvis.llm.base import LLMClient
from jarvis.llm.factory import create_llm_client
from jarvis.memory import ChromaVectorStore, InMemoryVectorStore, MemoryManager, SqliteStore
from jarvis.memory.vector_store import VectorStore
from jarvis.os_integration.base import Platform
from jarvis.os_integration.stub import StubPlatform
from jarvis.os_integration.windows import create_windows_platform
from jarvis.tools._jobs import BackgroundJobs
from jarvis.utils.errors import JarvisError, NativeUnavailableError
from jarvis.utils.logging import get_logger

log = get_logger(__name__)

MEMORY_DB_FILENAME: Final = "memory.sqlite3"
CHROMA_SUBDIR: Final = "chroma"
INTEGRATIONS_TIMEOUT: Final = httpx.Timeout(30.0, connect=10.0)


@runtime_checkable
class _Closeable(Protocol):
    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class Overrides:
    """Pre-built components that replace what the builders would create.

    Used by tests (fake LLM, scripted input) and by future front ends that
    bring their own input and output (the voice pipeline in step 7).
    """

    source: InputSource | None = None
    sink: OutputSink | None = None
    llm: LLMClient | None = None
    platform: Platform | None = None
    vectors: VectorStore | None = None
    registry: ToolRegistry | None = None


@dataclass
class Application:
    """The assembled assistant plus everything that must be closed afterwards.

    Attributes:
        orchestrator: Runs the conversation.
        agent: The agent the orchestrator drives, exposed for inspection.
        resources: Cleanup callbacks, run in reverse order of creation.
    """

    orchestrator: Orchestrator
    agent: Agent
    resources: AsyncExitStack = field(default_factory=AsyncExitStack)

    async def run(self) -> None:
        """Converse until the user leaves, then release every resource."""
        try:
            await self.orchestrator.run()
        finally:
            await self.resources.aclose()


async def build_application(settings: Settings, overrides: Overrides | None = None) -> Application:
    """Build the whole text-mode assistant.

    Raises:
        ConfigError: If the selected mode or provider lacks configuration.
        JarvisError: If a required component (the memory database) fails.
    """
    stack = AsyncExitStack()
    try:
        return await _assemble(settings, overrides or Overrides(), stack)
    except BaseException:
        await stack.aclose()
        raise


async def run_text_mode(settings: Settings, overrides: Overrides | None = None) -> None:
    """Build the assistant and converse in the terminal until the user exits."""
    application = await build_application(settings, overrides)
    await application.run()


async def _assemble(settings: Settings, overrides: Overrides, stack: AsyncExitStack) -> Application:
    if overrides.llm is None:
        settings.validate_for_mode()  # fail before any file or connection is opened
    bus = build_bus(stack)
    memory = await build_memory(settings, stack, vectors=overrides.vectors)
    llm = overrides.llm or build_llm(settings, stack)
    registry = overrides.registry or build_registry()
    platform = overrides.platform or build_platform(stack)
    integrations = build_integrations(settings, stack)
    source = overrides.source or TerminalInput()
    sink = overrides.sink or TerminalOutput(settings.assistant_name)
    ask = TerminalConfirmation(source, sink)
    services = ToolServices(platform, memory, integrations, BackgroundJobs())
    context = build_tool_context(settings, bus, ask, services)
    agent = build_agent(settings, llm, registry, context, memory)
    orchestrator = Orchestrator(agent=agent, state=StateMachine(bus), source=source, sink=sink)
    return Application(orchestrator=orchestrator, agent=agent, resources=stack)


def build_bus(stack: AsyncExitStack) -> EventBus:
    """The process-wide event bus, closed last so shutdown events still flow."""
    bus = EventBus()
    stack.push_async_callback(bus.close)
    return bus


async def build_memory(
    settings: Settings, stack: AsyncExitStack, *, vectors: VectorStore | None = None
) -> MemoryManager:
    """SQLite under ``data_dir`` plus a vector store, with a conversation started."""
    store = SqliteStore(settings.data_dir / MEMORY_DB_FILENAME)
    await stack.enter_async_context(store)
    manager = MemoryManager(
        store,
        vectors or await build_vector_store(settings, stack),
        recent_turns=settings.memory_recent_turns,
        semantic_hits=settings.memory_semantic_hits,
    )
    await manager.start_conversation()
    stack.push_async_callback(manager.end_conversation)
    return manager


def chroma_available() -> bool:
    """Whether the ``memory`` extra (ChromaDB) is installed."""
    return importlib.util.find_spec("chromadb") is not None


async def build_vector_store(settings: Settings, stack: AsyncExitStack) -> VectorStore:
    """ChromaDB when installed and openable, else the in-memory store with a warning."""
    if not chroma_available():
        log.warning("memory.vectors.in_memory", reason="chromadb is not installed")
        return InMemoryVectorStore()
    chroma = ChromaVectorStore(settings.data_dir / CHROMA_SUBDIR)
    try:
        await stack.enter_async_context(chroma)
    except JarvisError as error:
        log.warning("memory.vectors.in_memory", reason=str(error))
        return InMemoryVectorStore()
    log.info("memory.vectors.chroma", directory=str(settings.data_dir / CHROMA_SUBDIR))
    return chroma


def build_llm(settings: Settings, stack: AsyncExitStack) -> LLMClient:
    """The configured LLM client; its connection pool is closed at shutdown."""
    client = create_llm_client(settings)
    if isinstance(client, _Closeable):
        stack.push_async_callback(client.aclose)
    log.info("llm.selected", provider=settings.llm_provider, model=client.model)
    return client


def build_registry() -> ToolRegistry:
    """Discover every tool plugin; broken plugins are logged, not fatal."""
    registry = ToolRegistry.discover()
    for failure in registry.failures:
        log.warning("tools.plugin.skipped", module=failure.module, error=failure.error)
    log.info("tools.discovered", tools=list(registry.names))
    return registry


def build_platform(stack: AsyncExitStack) -> Platform:
    """``WindowsPlatform`` when ``jarvis_os`` loads, otherwise ``StubPlatform``."""
    platform: Platform
    try:
        platform = create_windows_platform()
    except NativeUnavailableError as error:
        log.warning("os.platform.stub", reason=str(error))
        platform = StubPlatform()
    stack.push_async_callback(platform.close)
    log.info("os.platform.selected", platform=type(platform).__name__)
    return platform


def build_integrations(settings: Settings, stack: AsyncExitStack) -> IntegrationServices:
    """External services for the tools: real providers or ``Stub*`` fallbacks.

    Also carries web search and YouTube transcripts for the web tools.
    """
    http = httpx.AsyncClient(timeout=INTEGRATIONS_TIMEOUT)
    stack.push_async_callback(http.aclose)
    return build_integration_services(settings, http)


def build_tool_context(
    settings: Settings,
    bus: EventBus,
    ask: ConfirmationAsker,
    services: ToolServices,
) -> ToolContext:
    """The services every tool call receives."""
    return ToolContext(
        settings=settings,
        bus=bus,
        ask_confirmation=ask,
        platform=services.platform,
        memory=services.memory,
        integrations=services.integrations,
        jobs=services.jobs,
    )


@dataclass(frozen=True, slots=True)
class ToolServices:
    """What tools reach through ``ToolContext``, built once per process.

    Grouped so the composition root passes one value instead of a growing
    argument list as later steps add services (vision, voice).
    """

    platform: Platform
    memory: MemoryManager
    integrations: IntegrationServices
    jobs: BackgroundJobs


def build_agent(
    settings: Settings,
    llm: LLMClient,
    registry: ToolRegistry,
    context: ToolContext,
    memory: MemoryManager,
) -> Agent:
    """The agent loop with persona, limits and the central confirmation policy."""
    return Agent(
        llm=llm,
        registry=registry,
        policy=ConfirmationPolicy.from_settings(settings),
        persona=Persona.from_settings(settings),
        context=context,
        memory=memory,
        limits=AgentLimits(max_tool_rounds=settings.max_tool_rounds),
        cloud_provider=settings.llm_provider == "anthropic",
    )


# Design notes:
# - One AsyncExitStack owns every cleanup. Callbacks run in reverse order,
#   so the conversation summary is written before SQLite closes and the
#   event bus closes last. A failure half-way through assembly closes what
#   was already opened before re-raising.
# - ToolContext.memory stays None on purpose: its step-6 placeholder
#   Protocol speaks free-text facts, while MemoryManager stores keyed facts.
#   Adapting one to the other would mean inventing keys; the memory tools
#   (step 8) replace the Protocol instead. The agent itself uses the real
#   MemoryManager for recall and recording.
# - The platform falls back to StubPlatform with a logged warning, never
#   silently: text-only development on a machine (or CI runner) without
#   the native build is a supported configuration, not an error.
# - ChromaDB failing to open degrades to the in-memory store for this
#   session (semantic recall is an index over SQLite, ADR-0008), while a
#   failing SQLite store is fatal: without it nothing can be remembered.
# - Wiring hooks for work on other branches that had not reached main when
#   this landed; each is a one-line change in the named builder:
#   * vision (feat/vision): in ``build_tool_context`` pass
#     ``vision=create_screen_reader(settings, platform)`` from
#     ``jarvis.vision.factory`` once ``ToolContext.vision`` exists.
#   * integration services (``integrations/services.py``): in
#     ``build_integrations`` call ``build_integration_services(settings, http)``
#     with the same shared ``http`` client and put the result on the context.
#   * provider builders (``core/providers.py``, LOCAL mode): ``build_llm``
#     swaps ``create_llm_client(settings)`` for ``providers.build_llm(settings)``;
#     ``build_stt``/``build_tts`` join the voice front end in step 7.
