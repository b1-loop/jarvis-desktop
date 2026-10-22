"""Composition root: builds every component from ``Settings`` and wires them together.

This is the one module (with ``__main__``, which only calls it, and the
front-end builder in ``voice_bootstrap``) that knows the concrete classes:
which LLM client, which vector store, which OS backend, which integration
providers. Each ``build_*`` function makes one choice, logs it, registers
the component's cleanup on an ``AsyncExitStack`` and returns the interface.
Everything downstream receives its dependencies by injection.

Assembly order follows the dependencies: settings -> event bus -> memory
-> LLM -> tool registry -> platform -> tool services (integrations, jobs,
vision) -> front end (terminal or voice) -> agent -> orchestrator ->
proactive scheduler -> dashboard and tray.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

import httpx

from jarvis.agent.confirmation import ConfirmationPolicy
from jarvis.agent.loop import Agent, AgentLimits
from jarvis.agent.persona import Persona
from jarvis.agent.registry import ToolRegistry
from jarvis.agent.tool import ConfirmationAsker, ToolContext
from jarvis.agent.tool_embeddings import EmbeddingToolRanker, ToolVectorCache
from jarvis.agent.tool_selection import ALL_TOOLS, RelevanceToolSelector, ToolSelector
from jarvis.config import Settings
from jarvis.core import providers
from jarvis.core.events import EventBus
from jarvis.core.orchestrator import (
    HELP_NOTICE,
    InputSource,
    Orchestrator,
    OutputSink,
    TerminalConfirmation,
    TerminalInput,
    TerminalOutput,
)
from jarvis.core.proactive_wiring import (
    BlockingBatteryProbe,
    IdleAnnouncer,
    MemoryReminderSource,
    ProactiveSources,
    ProbeBatterySource,
    RegistryToolInvoker,
    RoutineAwareTurns,
    build_trigger_runner,
    load_routine_registry,
)
from jarvis.core.state_machine import StateMachine
from jarvis.core.ui_wiring import AppControls, MicrophoneSwitch, build_dashboard, build_tray
from jarvis.integrations.base import ConsentRequiredError
from jarvis.integrations.consent import ConsentNotifier
from jarvis.integrations.services import IntegrationServices, build_integration_services
from jarvis.llm.base import LLMClient
from jarvis.llm.ollama_embeddings import OllamaEmbedder
from jarvis.memory import ChromaVectorStore, InMemoryVectorStore, MemoryManager, SqliteStore
from jarvis.memory.vector_store import VectorStore
from jarvis.os_integration.base import Platform
from jarvis.os_integration.errors import PlatformError
from jarvis.os_integration.stub import StubPlatform
from jarvis.os_integration.windows import create_windows_platform
from jarvis.proactive.routines import RoutineRegistry, RoutineRunner
from jarvis.proactive.scheduler import Scheduler
from jarvis.proactive.scheduler import SystemClock as SchedulerClock
from jarvis.tools._jobs import BackgroundJobs
from jarvis.tools.upcoming_events import CalendarTriggerSource
from jarvis.utils.errors import ConfigError, JarvisError, NativeUnavailableError
from jarvis.utils.logging import get_logger
from jarvis.vision.factory import create_screen_reader
from jarvis.vision.screen import ScreenReaderProtocol

log = get_logger(__name__)

MEMORY_DB_FILENAME: Final = "memory.sqlite3"
CHROMA_SUBDIR: Final = "chroma"
INTEGRATIONS_TIMEOUT: Final = httpx.Timeout(30.0, connect=10.0)
BATTERY_TOOL_MODULE: Final = "jarvis.tools.battery_status"
CONSENT_NOTIFICATION_TITLE: Final = "Jarvis: account not connected"
TOOL_VECTOR_CACHE: Final = Path("cache") / "tool_embeddings.json"
"""Embedded tool descriptions, under ``data_dir``, so a restart does not re-embed them."""
MS_PER_SECOND: Final = 1000


@runtime_checkable
class _Closeable(Protocol):
    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class FrontEndRequest:
    """What a front-end builder may use: the shared core built before it.

    Attributes:
        settings: The session's settings.
        bus: The process-wide event bus.
        state: The state machine the orchestrator will drive.
        platform: The OS backend (hotkeys).
        llm: The LLM client, which a front end may wrap (latency probes).
        stack: Where the front end registers its own cleanup.
    """

    settings: Settings
    bus: EventBus
    state: StateMachine
    platform: Platform
    llm: LLMClient
    stack: AsyncExitStack


@dataclass(frozen=True, slots=True)
class FrontEnd:
    """How the user talks to the assistant: terminal or voice.

    Attributes:
        source: Where requests come from.
        sink: Where answers go.
        ask: How confirmation questions are asked.
        llm: Replacement LLM client (a wrapper of the request's), if any.
        greeting: First notice of the session.
        attach: Called with the orchestrator once it exists (barge-in).
        toggle_overlay: Shows/hides the status overlay, if there is one.
        microphone: Mutes speech input, if there is any.
        bind_controls: Called with the tray's controls once they exist, so
            the interactive HUD's quit and mute go through them (ADR-0024).
    """

    source: InputSource
    sink: OutputSink
    ask: ConfirmationAsker
    llm: LLMClient | None = None
    greeting: str = HELP_NOTICE
    attach: Callable[[Orchestrator], None] | None = None
    toggle_overlay: Callable[[], None] | None = None
    microphone: MicrophoneSwitch | None = None
    bind_controls: Callable[[AppControls], None] | None = None


FrontEndBuilder = Callable[[FrontEndRequest], Awaitable[FrontEnd]]
"""Builds a front end from the shared core; the voice one lives in ``voice_bootstrap``."""


@dataclass(frozen=True, slots=True)
class Overrides:
    """Pre-built components that replace what the builders would create.

    Used by tests (fake LLM, scripted input) and by front ends that bring
    their own input and output (``front_end``, the voice pipeline).

    ``interactive_sign_in`` is the one launch choice here rather than a
    component: only a terminal session may open a browser or print a device
    code for Google or Microsoft consent, so the default is ``False`` and
    the console ``--text-only`` launch alone turns it on.
    """

    source: InputSource | None = None
    sink: OutputSink | None = None
    llm: LLMClient | None = None
    platform: Platform | None = None
    vectors: VectorStore | None = None
    registry: ToolRegistry | None = None
    front_end: FrontEndBuilder | None = None
    interactive_sign_in: bool = False


@dataclass(frozen=True, slots=True)
class ToolServices:
    """What tools reach through ``ToolContext``, built once per process.

    Grouped so the composition root passes one value instead of a growing
    argument list as services are added.
    """

    platform: Platform
    memory: MemoryManager
    integrations: IntegrationServices
    jobs: BackgroundJobs
    vision: ScreenReaderProtocol | None = None


@dataclass
class Application:
    """The assembled assistant plus everything that must be closed afterwards.

    Attributes:
        orchestrator: Runs the conversation.
        agent: The agent the orchestrator drives, exposed for inspection.
        resources: Cleanup callbacks, run in reverse order of creation.
        controls: The tray menu's actions (show dashboard, quit); the
            windowless app reuses them when a second launch asks the
            running one to come forward.
    """

    orchestrator: Orchestrator
    agent: Agent
    resources: AsyncExitStack = field(default_factory=AsyncExitStack)
    controls: AppControls | None = None
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _stop_requested: bool = field(default=False, init=False, repr=False)

    async def run(self) -> None:
        """Converse until the user leaves (or quits from the tray), then release everything."""
        self._task = asyncio.current_task()
        try:
            await self.orchestrator.run()
        except asyncio.CancelledError:
            if not self._stop_requested or self._task is None:
                raise
            self._task.uncancel()
            log.info("application.stopped", reason="quit requested")
        finally:
            await self.resources.aclose()

    def request_stop(self) -> None:
        """End the session from outside the conversation (tray "Quit")."""
        self._stop_requested = True
        if self._task is not None:
            self._task.cancel()


async def build_application(settings: Settings, overrides: Overrides | None = None) -> Application:
    """Build the whole assistant; the front end is the terminal unless overridden.

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
    store = await build_store(settings, stack)
    memory = await build_memory(settings, stack, vectors=overrides.vectors, store=store)
    llm = overrides.llm or build_llm(settings, stack)
    registry = overrides.registry or build_registry()
    platform = overrides.platform or build_platform(stack)
    services = build_tool_services(
        settings, stack, platform, memory, interactive_sign_in=overrides.interactive_sign_in
    )
    state = StateMachine(bus)
    request = FrontEndRequest(settings, bus, state, platform, llm, stack)
    front = await (overrides.front_end or _text_front_end(overrides))(request)
    context = build_tool_context(settings, bus, front.ask, services)
    selector = build_tool_selector(settings, stack, registry)
    agent = build_agent(settings, front.llm or llm, registry, context, memory, selector=selector)
    routines = build_routines(settings)
    invoker = RegistryToolInvoker(registry, agent.policy, context)
    orchestrator = Orchestrator(
        agent=RoutineAwareTurns(agent, routines, invoker),
        state=state,
        source=front.source,
        sink=front.sink,
        greeting=front.greeting,
    )
    if front.attach is not None:
        front.attach(orchestrator)
    application = Application(orchestrator=orchestrator, agent=agent, resources=stack)
    await build_proactive(
        settings,
        stack,
        bus=bus,
        sink=front.sink,
        services=services,
        routines=routines,
        invoker=invoker,
    )
    application.controls = await build_ui(
        settings, stack, bus=bus, store=store, application=application, front=front
    )
    return application


def _text_front_end(overrides: Overrides) -> FrontEndBuilder:
    async def build(request: FrontEndRequest) -> FrontEnd:
        return build_text_front_end(request.settings, overrides)

    return build


def build_text_front_end(settings: Settings, overrides: Overrides | None = None) -> FrontEnd:
    """Terminal input and output, with y/n confirmation; scripted ones from ``overrides``."""
    overrides = overrides or Overrides()
    source = overrides.source or TerminalInput()
    sink = overrides.sink or TerminalOutput(settings.assistant_name)
    return FrontEnd(source=source, sink=sink, ask=TerminalConfirmation(source, sink))


def build_bus(stack: AsyncExitStack) -> EventBus:
    """The process-wide event bus, closed last so shutdown events still flow."""
    bus = EventBus()
    stack.push_async_callback(bus.close)
    return bus


async def build_store(settings: Settings, stack: AsyncExitStack) -> SqliteStore:
    """The SQLite store under ``data_dir``; memory and the dashboard's inspector share it."""
    store = SqliteStore(settings.data_dir / MEMORY_DB_FILENAME)
    await stack.enter_async_context(store)
    return store


async def build_memory(
    settings: Settings,
    stack: AsyncExitStack,
    *,
    vectors: VectorStore | None = None,
    store: SqliteStore | None = None,
) -> MemoryManager:
    """SQLite plus a vector store, with a conversation started."""
    manager = MemoryManager(
        store or await build_store(settings, stack),
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
    """The LLM for the mode and provider; its connection pool is closed at shutdown."""
    client = providers.build_llm(settings)
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


def build_integrations(
    settings: Settings,
    stack: AsyncExitStack,
    *,
    platform: Platform | None = None,
    interactive_sign_in: bool = False,
) -> IntegrationServices:
    """External services for the tools: real providers or ``Stub*`` fallbacks.

    Also carries web search and YouTube transcripts for the web tools.
    Without ``interactive_sign_in`` an account that needs consent raises
    "run connect_accounts.py" instead of opening a flow, and the first such
    error per account is shown as a notification on ``platform``.
    """
    http = httpx.AsyncClient(timeout=INTEGRATIONS_TIMEOUT)
    stack.push_async_callback(http.aclose)
    notifier = None
    if not interactive_sign_in and platform is not None:
        notifier = consent_notifier(platform)
    return build_integration_services(
        settings, http, interactive_sign_in=interactive_sign_in, consent_notifier=notifier
    )


def consent_notifier(platform: Platform) -> ConsentNotifier:
    """Shows "Google isn't connected yet; run ..." as a system notification."""

    async def notify(error: ConsentRequiredError) -> None:
        try:
            await platform.notify(CONSENT_NOTIFICATION_TITLE, error.remedy)
        except PlatformError:
            # The spoken answer still carries the remedy; a lost balloon is not fatal.
            log.exception("integrations.consent.notify_failed", provider=error.provider)

    return notify


def build_jobs(stack: AsyncExitStack) -> BackgroundJobs:
    """The one tracker of background processes (tools start, the proactive trigger reports)."""
    jobs = BackgroundJobs()
    stack.push_async_callback(jobs.close)
    return jobs


def build_vision(settings: Settings, platform: Platform) -> ScreenReaderProtocol | None:
    """Screen understanding, or ``None`` with a warning when the provider cannot do it."""
    try:
        reader = create_screen_reader(settings, platform)
    except ConfigError as error:
        log.warning("vision.unavailable", reason=str(error))
        return None
    log.info("vision.ready", provider=settings.llm_provider)
    return reader


def build_tool_services(
    settings: Settings,
    stack: AsyncExitStack,
    platform: Platform,
    memory: MemoryManager,
    *,
    interactive_sign_in: bool = False,
) -> ToolServices:
    """Everything tools reach through their context, built once."""
    integrations = build_integrations(
        settings, stack, platform=platform, interactive_sign_in=interactive_sign_in
    )
    return ToolServices(
        platform=platform,
        memory=memory,
        integrations=integrations,
        jobs=build_jobs(stack),
        vision=build_vision(settings, platform),
    )


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
        vision=services.vision,
        jobs=services.jobs,
    )


def build_tool_selector(
    settings: Settings, stack: AsyncExitStack, registry: ToolRegistry
) -> ToolSelector:
    """Hybrid per-turn tool selection; words only without an embedding model (ADR-0020).

    The embedding ranker starts embedding the tool descriptions in the
    background right away, so the first turns may rank by words alone.
    """
    model = settings.tool_embedding_model
    if model is None or settings.tools_per_turn() == ALL_TOOLS:
        log.info("tool_selection.ready", semantic=False, max_tools=settings.tools_per_turn())
        return RelevanceToolSelector.from_settings(settings)
    embedder = OllamaEmbedder.from_settings(settings, model)
    stack.push_async_callback(embedder.aclose)
    ranker = EmbeddingToolRanker(
        embedder,
        cache=ToolVectorCache(settings.data_dir / TOOL_VECTOR_CACHE),
        budget_seconds=settings.tool_embedding_timeout_ms / MS_PER_SECOND,
    )
    stack.push_async_callback(ranker.aclose)
    ranker.warm_up(registry.tools())
    log.info(
        "tool_selection.ready", semantic=True, model=model, max_tools=settings.tools_per_turn()
    )
    return RelevanceToolSelector.from_settings(settings, ranker)


def build_agent(
    settings: Settings,
    llm: LLMClient,
    registry: ToolRegistry,
    context: ToolContext,
    memory: MemoryManager,
    *,
    selector: ToolSelector | None = None,
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
        selector=selector,
    )


def build_routines(settings: Settings) -> RoutineRegistry:
    """Routines from ``routines_dir``; a missing directory means none."""
    registry = load_routine_registry(settings.routines_dir)
    log.info("proactive.routines.ready", routines=[r.name for r in registry.routines])
    return registry


def build_battery_source() -> ProbeBatterySource | None:
    """Battery readings via the battery tool's psutil probe; ``None`` without psutil."""
    try:
        module = importlib.import_module(BATTERY_TOOL_MODULE)
    except ImportError as error:
        log.warning("proactive.battery.unavailable", reason=str(error))
        return None
    probe: BlockingBatteryProbe = module.PsutilBatteryProbe()
    return ProbeBatterySource(probe)


def build_proactive_sources(services: ToolServices) -> ProactiveSources:
    """What the standard triggers watch, from the same services the tools use."""
    return ProactiveSources(
        battery=build_battery_source(),
        reminders=MemoryReminderSource(services.memory),
        commands=services.jobs,
        calendar=CalendarTriggerSource(services.integrations.calendar),
    )


async def build_proactive(
    settings: Settings,
    stack: AsyncExitStack,
    *,
    bus: EventBus,
    sink: OutputSink,
    services: ToolServices,
    routines: RoutineRegistry,
    invoker: RegistryToolInvoker,
) -> Scheduler | None:
    """Start the scheduler with the standard triggers and the timed routines.

    Notifications and timed routines speak through the front end's sink
    once the assistant is idle, and also appear as system notifications.
    """
    if not settings.proactive_enabled:
        log.info("proactive.disabled")
        return None
    clock = SchedulerClock()
    scheduler = Scheduler(clock)
    announcer = IdleAnnouncer(sink, bus, services.platform.notify)
    stack.callback(announcer.close)
    poll = timedelta(seconds=settings.proactive_poll_seconds)
    build_trigger_runner(
        scheduler, announcer, build_proactive_sources(services), clock=clock, poll=poll
    )
    routines.schedule(scheduler, RoutineRunner(invoker, announcer))
    await scheduler.start()
    stack.push_async_callback(scheduler.stop)
    return scheduler


async def build_ui(
    settings: Settings,
    stack: AsyncExitStack,
    *,
    bus: EventBus,
    store: SqliteStore,
    application: Application,
    front: FrontEnd,
) -> AppControls:
    """Dashboard and tray icon (or their stubs), with the tray menu wired to the session."""
    dashboard = await build_dashboard(settings, bus, stack, memory=store)
    controls = AppControls(
        stop=application.request_stop,
        dashboard=dashboard,
        toggle_overlay=front.toggle_overlay,
        microphone=front.microphone,
    )
    await build_tray(settings, bus, stack, controls)
    if front.bind_controls is not None:
        front.bind_controls(controls)
    return controls


# Design notes:
# - One AsyncExitStack owns every cleanup. Callbacks run in reverse order,
#   so the tray, dashboard and scheduler stop first, the front end next, the
#   conversation summary is written before SQLite closes, and the event bus
#   closes last. A failure half-way through assembly closes what was
#   already opened before re-raising.
# - The front end is a builder, not a pair of objects, because the voice
#   front end needs the bus, state machine and platform built here and
#   hands back more than input and output (confirmation, an LLM latency
#   probe, barge-in wiring, overlay and microphone controls).
# - The platform falls back to StubPlatform with a logged warning, never
#   silently: text-only development on a machine (or CI runner) without
#   the native build is a supported configuration, not an error.
# - ChromaDB failing to open degrades to the in-memory store for this
#   session (semantic recall is an index over SQLite, ADR-0008), while a
#   failing SQLite store is fatal: without it nothing can be remembered.
# - Vision degrades to None (the screen tools say so) when the provider
#   cannot see images or has no key; the rest of the assistant is unaffected.
# - The battery probe is imported by name because psutil is an optional
#   extra; without it the battery trigger is skipped with a warning.
# - Account sign-in is non-interactive unless Overrides says otherwise: the
#   voice front end and the windowless app have no terminal to show a
#   browser prompt or device code in, so they report "run
#   connect_accounts.py" (spoken, plus one notification per account) and
#   only the console --text-only launch may start a consent flow.
# - Speech providers (providers.build_stt/build_tts) are built only by the
#   voice front end: --text-only must not need jarvis_audio, which edge-tts
#   decoding requires.
