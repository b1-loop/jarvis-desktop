"""Measure end-to-end voice latency per pipeline stage, without a human.

Usage::

    uv run python scripts/benchmark_latency.py                    # cloud-like and local, silent
    uv run python scripts/benchmark_latency.py --modes cloud --runs 10
    uv run python scripts/benchmark_latency.py --sink device      # play the answers aloud

Each prompt is synthesised once with edge-tts (cached as a WAV under
``data/bench/prompts``) and then *spoken into the real pipeline*: a
``jarvis_audio.ReplayFeed`` replaces the microphone, so the native engine's
ring buffer, Silero VAD, segmenter, wake-word model and native barge-in run
exactly as they do on a live microphone, at real-time pace. Everything after
that is the real application from ``core.bootstrap``: STT, the agent with the
configured LLM and every tool plugin, memory and TTS. Only
the operating system is a ``StubPlatform`` (its hotkey starts each turn, so
no wake word has to be synthesised) and the overlay, tray, dashboard and
proactive scheduler are off.

Playback goes to a null sink by default: it accepts chunks and "plays"
them at the real-time rate without sound, so a benchmark never talks over
the person at the computer. Every stage up to "first audio chunk handed to
the device" is measured identically either way; the real device adds one
period of its buffer (about 10 ms) before the sound is audible.
``--sink device`` uses the real ``jarvis_audio.Playback``.

Stages, all from the same run:

* end of speech -> transcript, transcript -> first token, first sentence ->
  first audio: the ``LatencySample`` events the pipeline publishes;
* end of speech -> first audio: from the moment Python learns of
  ``SPEECH_ENDED`` to the first answer samples handed to the device, which
  also covers the time the LLM needs to finish its first sentence;
* last word -> first audio: the same plus the VAD's silence hangover, i.e.
  what the user waits for after they stop talking;
* barge-in: a second prompt is replayed while Jarvis is speaking; speech
  onset -> native playback stop (engine sample clock) and the moment Python
  sees ``SPEECH_STARTED`` -> ``LISTENING``.

Modes: ``cloud`` (faster-whisper + Ollama + edge-tts, the project default;
Ollama is local but the TTS needs the network) and ``local`` (whisper.cpp +
llama.cpp + Piper). Each prompt gets one warm-up run, then ``--runs``
measured runs; the report gives median and p90, prints a Markdown table and
writes ``data/bench/results.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import wave
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from jarvis import _native
from jarvis.audio.engine import create_native_playback
from jarvis.config import RuntimeMode, Settings
from jarvis.core.bootstrap import Overrides, build_application
from jarvis.core.events import (
    Event,
    LatencySample,
    StateChanged,
    ToolCallStarted,
    TranscriptReady,
)
from jarvis.core.voice_bootstrap import VoiceFactories, VoiceFrontEndBuilder
from jarvis.os_integration.stub import StubPlatform
from jarvis.stt.faster_whisper import resample_linear
from jarvis.tts.edge_tts import EdgeTTS
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import configure_logging

SAMPLE_RATE: Final = 16_000
BENCH_DIR: Final = Path("data/bench")
PROMPT_DIR: Final = BENCH_DIR / "prompts"
STATE_DIR: Final = BENCH_DIR / "state"
RESULTS_FILE: Final = BENCH_DIR / "results.json"
SPEECH_FLOOR: Final = 0.01
"""Amplitude below which a synthesised prompt counts as silence (trimming)."""
LEAD_IN_SECONDS: Final = 0.2
TURN_TIMEOUT_SECONDS: Final = 180.0
BARGE_IN_DELAY_SECONDS: Final = 1.0
"""How long after the answer becomes audible the interruption starts (beyond the echo guard)."""
PRESS_ATTEMPTS: Final = 10
PRESS_RETRY_SECONDS: Final = 0.5
MS: Final = 1000.0
P90: Final = 0.9
JSON_SCHEMA_VERSION: Final = 1

Samples = NDArray[np.float32]


# --- prompts --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Prompt:
    """One fixed request, spoken by an edge-tts voice that is not Jarvis's own."""

    key: str
    language: str
    text: str


PROMPT_VOICES: Final = {"en": "en-US-AriaNeural", "sv": "sv-SE-SofieNeural"}
JARVIS_VOICES: Final = {"en": "en-GB-RyanNeural", "sv": "sv-SE-MattiasNeural"}
PROMPTS: Final = {
    "en": (
        Prompt("en-capital", "en", "What is the capital of Australia?"),
        Prompt("en-time", "en", "What time is it right now?"),
        Prompt("en-tip", "en", "Give me one tip for staying focused while I work."),
    ),
    "sv": (
        Prompt("sv-capital", "sv", "Vad är huvudstaden i Australien?"),
        Prompt("sv-time", "sv", "Vad är klockan just nu?"),
    ),
}
BARGE_IN_ANSWER: Final = Prompt(
    "en-story", "en", "Tell me a story about a lighthouse keeper, in about eight sentences."
)
BARGE_IN_INTERRUPTION: Final = Prompt("en-hold-on", "en", "Hold on what is two plus two")
"""No punctuation on purpose: a lone "Stop." followed by a pause is shorter
than the segmenter's minimum speech duration and is (correctly) ignored."""


@dataclass(frozen=True, slots=True)
class PromptAudio:
    """A prompt as 16 kHz samples, plus where its first word starts and its last word ends."""

    prompt: Prompt
    samples: Samples
    speech_start_s: float
    speech_end_s: float


def speech_bounds(samples: Samples, *, floor: float = SPEECH_FLOOR) -> tuple[float, float]:
    """Seconds to the first audible sample and to the end of the last one."""
    loud = np.flatnonzero(np.abs(samples) > floor)
    if loud.size == 0:
        return 0.0, 0.0
    return float(loud[0]) / SAMPLE_RATE, float(loud[-1] + 1) / SAMPLE_RATE


def write_wav(path: Path, samples: Samples) -> None:
    """Store mono float32 samples as 16-bit PCM at 16 kHz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * np.iinfo(np.int16).max).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


def read_wav(path: Path) -> Samples:
    """Read a file written by :func:`write_wav`."""
    with wave.open(str(path), "rb") as wav:
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    return (pcm.astype(np.float32) / np.iinfo(np.int16).max).astype(np.float32)


def prompt_path(prompt: Prompt) -> Path:
    """Cache file for ``prompt``; the name changes with the voice or the text."""
    voice = PROMPT_VOICES[prompt.language]
    digest = hashlib.sha256(f"{voice}|{prompt.text}".encode()).hexdigest()[:12]
    return PROMPT_DIR / f"{prompt.key}-{digest}.wav"


async def synthesise(prompt: Prompt) -> Samples:
    """Speak ``prompt`` with edge-tts and resample it to the engine's 16 kHz."""
    tts = EdgeTTS.from_settings(Settings(tts_voice=PROMPT_VOICES[prompt.language]))
    chunks = [chunk.samples async for chunk in tts.synthesize(prompt.text)]
    speech = resample_linear(np.concatenate(chunks), tts.sample_rate, SAMPLE_RATE)
    lead_in = np.zeros(int(LEAD_IN_SECONDS * SAMPLE_RATE), dtype=np.float32)
    return np.concatenate([lead_in, speech]).astype(np.float32)


async def load_prompt(prompt: Prompt) -> PromptAudio:
    """The cached WAV, synthesising it on first use."""
    path = prompt_path(prompt)
    if not path.exists():
        write_wav(path, await synthesise(prompt))
    samples = read_wav(path)
    return PromptAudio(prompt, samples, *speech_bounds(samples))


# --- statistics and reporting (pure) ---------------------------------------------


@dataclass(frozen=True, slots=True)
class Stage:
    """A row of the results table."""

    key: str
    label: str


HOTKEY_TO_LISTENING: Final = Stage("hotkey_to_listening", "Hotkey -> listening (earcon)")
SPEECH_END_TO_TRANSCRIPT: Final = Stage("speech_end_to_transcript", "End of speech -> transcript")
TRANSCRIPT_TO_TOKEN: Final = Stage("transcript_to_first_token", "Transcript -> first LLM token")
SENTENCE_TO_AUDIO: Final = Stage("first_sentence_to_audio", "First sentence -> first audio")
SPEECH_END_TO_AUDIO: Final = Stage(
    "speech_end_to_first_audio", "**End of speech -> first audio (total)**"
)
LAST_WORD_TO_AUDIO: Final = Stage(
    "last_word_to_first_audio", "Last word -> first audio (incl. VAD hangover)"
)
BARGE_IN_TO_STOP: Final = Stage(
    "barge_in_onset_to_stop", "Barge-in: speech onset -> playback stopped (native)"
)
BARGE_IN_TO_LISTENING: Final = Stage(
    "barge_in_to_listening", "Barge-in: playback stopped -> LISTENING (Python)"
)
STAGES: Final = (
    HOTKEY_TO_LISTENING,
    SPEECH_END_TO_TRANSCRIPT,
    TRANSCRIPT_TO_TOKEN,
    SENTENCE_TO_AUDIO,
    SPEECH_END_TO_AUDIO,
    LAST_WORD_TO_AUDIO,
    BARGE_IN_TO_STOP,
    BARGE_IN_TO_LISTENING,
)
SAMPLE_STAGES: Final = {
    "voice.wake_to_listening": HOTKEY_TO_LISTENING.key,
    "voice.speech_end_to_transcript": SPEECH_END_TO_TRANSCRIPT.key,
    "voice.transcript_to_first_token": TRANSCRIPT_TO_TOKEN.key,
    "voice.first_sentence_to_audio": SENTENCE_TO_AUDIO.key,
}
"""``LatencySample`` stage names (``core/voice_latency.py``) -> table rows."""


@dataclass(frozen=True, slots=True)
class StageStats:
    """Summary of one stage's samples, in milliseconds."""

    n: int
    median_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile (numpy's default method); ``fraction`` in [0, 1]."""
    if not values:
        raise ValueError("percentile of no values")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be within [0, 1], got {fraction}")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarise(values: Sequence[float]) -> StageStats | None:
    """Median, p90, min and max, or ``None`` when the stage was not measured."""
    if not values:
        return None
    return StageStats(
        n=len(values),
        median_ms=statistics.median(values),
        p90_ms=percentile(values, P90),
        min_ms=min(values),
        max_ms=max(values),
    )


@dataclass
class ScenarioResult:
    """Every measured run of one mode and language."""

    mode: str
    language: str
    providers: dict[str, str]
    runs: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Column header, e.g. ``cloud / en``."""
        return f"{self.mode} / {self.language}"

    def values(self, stage: Stage) -> list[float]:
        """Every measured value of ``stage`` across this scenario's runs."""
        return [float(run[stage.key]) for run in self.runs if stage.key in run]

    def turns(self) -> list[dict[str, Any]]:
        """The measured request turns (barge-in runs excluded)."""
        return [run for run in self.runs if "transcript" in run]

    def tool_turns(self) -> int:
        """Measured turns in which the model called at least one tool."""
        return sum(1 for run in self.turns() if run.get("tools"))

    def summary(self) -> dict[str, StageStats | None]:
        """Stage key -> statistics."""
        return {stage.key: summarise(self.values(stage)) for stage in STAGES}


def format_cell(stats: StageStats | None) -> str:
    """``median / p90`` in whole milliseconds, or a dash."""
    if stats is None:
        return "-"
    return f"{stats.median_ms:.0f} / {stats.p90_ms:.0f}"


def render_table(scenarios: Sequence[ScenarioResult]) -> str:
    """Markdown table: one row per stage, one ``median / p90 ms`` column per scenario."""
    header = "| Stage | " + " | ".join(f"{s.label} (ms)" for s in scenarios) + " |"
    divider = "| --- |" + " ---: |" * len(scenarios)
    summaries = [scenario.summary() for scenario in scenarios]
    rows = [
        f"| {stage.label} | " + " | ".join(format_cell(s[stage.key]) for s in summaries) + " |"
        for stage in STAGES
    ]
    counts = (
        "| Measured turns (of which called a tool) | "
        + " | ".join(f"{len(s.turns())} ({s.tool_turns()})" for s in scenarios)
        + " |"
    )
    return "\n".join([header, divider, *rows, counts])


def report_json(
    scenarios: Sequence[ScenarioResult], *, hardware: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """The document written to ``data/bench/results.json``."""
    return {
        "schema": JSON_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "hardware": hardware,
        "config": config,
        "stages": {stage.key: stage.label.strip("*") for stage in STAGES},
        "scenarios": [
            {
                "mode": scenario.mode,
                "language": scenario.language,
                "providers": scenario.providers,
                "summary": {
                    key: None if stats is None else asdict(stats)
                    for key, stats in scenario.summary().items()
                },
                "runs": scenario.runs,
                "errors": scenario.errors,
            }
            for scenario in scenarios
        ],
    }


def hardware_info() -> dict[str, Any]:
    """CPU, cores, RAM, GPU and OS of this machine, for the table's caption."""
    info: dict[str, Any] = {
        "cpu": cpu_name(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "gpu": gpu_name(),
    }
    with contextlib.suppress(ImportError):
        psutil = importlib.import_module("psutil")  # optional: the desktop extra
        info["cores"] = psutil.cpu_count(logical=False)
        info["threads"] = psutil.cpu_count()
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30)
    return info


def cpu_name() -> str:
    """Readable CPU model (the registry on Windows, ``platform`` elsewhere)."""
    if sys.platform != "win32":
        return platform.processor() or platform.machine()
    winreg = importlib.import_module("winreg")
    key = winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
    )
    return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()


def gpu_name() -> str | None:
    """The NVIDIA GPU, if ``nvidia-smi`` exists; Ollama uses it, faster-whisper may not."""
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return None
    query = [tool, "--query-gpu=name", "--format=csv,noheader"]
    result = subprocess.run(query, capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def caption(hardware: dict[str, Any], runs: int, sink: str) -> str:
    """One line under the table naming the machine and the method."""
    cores = f"{hardware.get('cores', '?')} cores / {hardware.get('threads', '?')} threads"
    gpu = hardware.get("gpu") or "no NVIDIA GPU"
    return (
        f"{hardware['cpu']} ({cores}), {hardware.get('ram_gb', '?')} GB RAM, {gpu}, "
        f"{hardware['os']}. Median / p90 of {runs} runs per prompt after one warm-up; "
        f"playback sink: {sink}."
    )


# --- instrumentation ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Mark:
    """Something that happened during a run, with its ``perf_counter`` time."""

    at: float
    kind: str
    value: Any = None


class Recorder:
    """An append-only timeline the benchmark waits on and measures from.

    Every writer runs on the event loop thread (bus callbacks, the engine
    poller, the speaker), so a plain list and an ``asyncio.Event`` suffice.
    """

    def __init__(self) -> None:
        self.marks: list[Mark] = []
        self._changed = asyncio.Event()

    def add(self, kind: str, value: Any = None) -> Mark:
        """Record ``kind`` now and wake every waiter."""
        mark = Mark(time.perf_counter(), kind, value)
        self.marks.append(mark)
        self._changed.set()
        return mark

    def position(self) -> int:
        """Index to search from, for "after this point" queries."""
        return len(self.marks)

    def find(self, kind: str, start: int, test: Callable[[Any], bool] | None = None) -> Mark | None:
        """First mark of ``kind`` at or after ``start`` whose value passes ``test``."""
        for mark in self.marks[start:]:
            if mark.kind == kind and (test is None or test(mark.value)):
                return mark
        return None

    async def wait(self, kind: str, start: int, test: Callable[[Any], bool] | None = None) -> Mark:
        """Wait (bounded by the turn timeout) for :meth:`find` to succeed."""
        async with asyncio.timeout(TURN_TIMEOUT_SECONDS):
            while (mark := self.find(kind, start, test)) is None:
                self._changed.clear()
                await self._changed.wait()
        return mark


class TimedPlayback:
    """``NativePlayback`` decorator that records when audio reaches the device."""

    def __init__(self, inner: Any, recorder: Recorder) -> None:
        self.inner = inner
        self._recorder = recorder

    def enqueue(self, samples: Samples) -> None:
        """Forward, marking the moment as ``enqueue``."""
        self._recorder.add("enqueue", samples.size)
        self.inner.enqueue(samples)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class NullPlayback:
    """A ``NativePlayback`` that renders into nothing at the real-time rate.

    For headless runs without an output device. Its barge-in stop is
    emulated in Python by :class:`ReplayEngine`, which adds up to one poll
    interval (10 ms) compared with the native stop.
    """

    def __init__(self, sample_rate: int, clock: Callable[[], float] = time.perf_counter) -> None:
        self._rate = sample_rate
        self._clock = clock
        self._ends_at = 0.0
        self._stream_open = False
        self._stops = 0

    def enqueue(self, samples: Samples) -> None:
        """Account for ``samples`` as if they were played after what is queued."""
        start = max(self._clock(), self._ends_at)
        self._ends_at = start + samples.size / self._rate
        self._stream_open = True

    def stop(self) -> None:
        """Discard everything and count the stop, like ``Playback::stop``."""
        self._ends_at = self._clock()
        self._stream_open = False
        self._stops += 1

    def mark_end_of_stream(self) -> None:
        """No more chunks follow for this utterance."""
        self._stream_open = False

    def is_playing(self) -> bool:
        """True while audio is "playing" or more is expected."""
        return self._stream_open or self._clock() < self._ends_at

    def queued_samples(self) -> int:
        """Samples not yet "rendered"."""
        return max(0, round((self._ends_at - self._clock()) * self._rate))

    def stop_count(self) -> int:
        """Stops so far."""
        return self._stops

    def output_level(self) -> float:
        """Always silent."""
        return 0.0

    def sample_rate(self) -> int:
        """The rate chunks arrive at."""
        return self._rate


class ReplayEngine:
    """``NativeEngine`` decorator: records events and arms barge-in on the real playback."""

    def __init__(self, inner: Any, recorder: Recorder) -> None:
        self.inner = inner
        self._recorder = recorder
        self._python_target: NullPlayback | None = None

    def poll_events(self) -> list[Any]:
        """Forward, marking each event as ``audio`` when Python sees it."""
        events: list[Any] = self.inner.poll_events()
        for event in events:
            self._recorder.add("audio", (event.type.name, int(event.sample_position)))
            if event.type.name == "SPEECH_STARTED" and self._python_target is not None:
                self._python_target.stop()
        return events

    def set_barge_in_target(self, playback: Any) -> None:
        """Hand the real playback to the engine; emulate the stop for the null sink."""
        if isinstance(playback, TimedPlayback):
            playback = playback.inner
        if isinstance(playback, NullPlayback):
            self._python_target = playback
            self.inner.set_barge_in_target(None)
            return
        self._python_target = None
        self.inner.set_barge_in_target(playback)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


# --- one session: the real application on a replayed microphone ------------------------


@dataclass(frozen=True, slots=True)
class SessionSpec:
    """Which providers and language one application is built with."""

    mode: str
    language: str
    sink: str


class Session:
    """Builds the voice application for a :class:`SessionSpec` and drives turns."""

    def __init__(self, spec: SessionSpec) -> None:
        self.spec = spec
        self.recorder = Recorder()
        self.feed: Any = None
        self.settings: Any = None
        self._platform: Any = None
        self._state: Any = None
        self._application: Any = None
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> Session:
        self.feed = _native.load("jarvis_audio").ReplayFeed()
        self.settings = session_settings(self.spec)
        self._application = await self._build()
        self._task = asyncio.create_task(self._application.run(), name="bench-app")
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._application is not None and self._task is not None:
            self._application.request_stop()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _build(self) -> Any:
        self._platform = StubPlatform()
        builder = VoiceFrontEndBuilder(factories=self._factories(), write=lambda _text: None)

        async def front_end(request: Any) -> Any:
            request.bus.subscribe_callback(Event, self._on_event)
            self._state = request.state
            return await builder(request)

        overrides = Overrides(platform=self._platform, front_end=front_end)
        application = await build_application(self.settings, overrides)
        self.recorder.add("state", ("", self._state.state.value))
        return application

    def _factories(self) -> Any:
        def engine(paths: Any, *, wake_word: bool, min_speech_ms: int | None) -> ReplayEngine:
            native = replay_engine(paths, self.feed, wake_word, min_speech_ms)
            return ReplayEngine(native, self.recorder)

        return VoiceFactories(engine=engine, playback=self._playback)

    def _playback(self, rate: int) -> TimedPlayback:
        if self.spec.sink == "null":
            return TimedPlayback(NullPlayback(rate), self.recorder)
        return TimedPlayback(create_native_playback(rate), self.recorder)

    def _on_event(self, event: Event) -> None:
        if isinstance(event, StateChanged):
            self.recorder.add("state", (event.reason, event.new.value))
        elif isinstance(event, LatencySample):
            self.recorder.add("sample", (event.stage, event.ms))
        elif isinstance(event, TranscriptReady):
            self.recorder.add("transcript", event.text)
        elif isinstance(event, ToolCallStarted):
            self.recorder.add("tool", event.name)

    @property
    def state(self) -> str:
        """The assistant's current state, e.g. ``"speaking"``."""
        return str(self._state.state.value)

    async def start_turn(self, audio: PromptAudio) -> tuple[int, float]:
        """Press the hotkey, wait for LISTENING and speak ``audio`` into the replay feed."""
        if self.state != "idle":
            await self.recorder.wait("state", self.recorder.position(), lambda v: v[1] == "idle")
        start = self.recorder.position()
        await self._press_until_listening(start)
        spoken_at = time.perf_counter()
        self.feed.push(audio.samples)
        return start, spoken_at

    async def _check_alive(self) -> int:
        """The hotkey id, or the reason the application under test stopped."""
        if self._task is not None and self._platform.hotkeys and not self._task.done():
            return int(next(iter(self._platform.hotkeys)))
        # Shutting down (the hotkey is released first) or already gone.
        if self._task is not None:
            await asyncio.wait([self._task], timeout=TURN_TIMEOUT_SECONDS)
        done = self._task is not None and self._task.done() and not self._task.cancelled()
        error = self._task.exception() if done and self._task is not None else None
        raise ApplicationStoppedError(f"the application stopped: {error!r}") from error

    async def _press_until_listening(self, start: int) -> None:
        # The hotkey trigger drops presses made while the voice input was
        # still finishing the previous turn, so a press can race it; repeat.
        for _ in range(PRESS_ATTEMPTS):
            self._platform.press(await self._check_alive())
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(PRESS_RETRY_SECONDS):
                    await self.recorder.wait("state", start, lambda v: v[1] == "listening")
                    return
        raise BenchmarkError("the hotkey did not start a turn")

    async def finish_turn(self, start: int) -> None:
        """Wait until the turn has been answered (or abandoned) and Jarvis is idle again."""
        await self.recorder.wait("state", start, lambda v: v[1] == "idle")

    async def turn(self, audio: PromptAudio) -> dict[str, Any]:
        """One request from hotkey to the end of the spoken answer."""
        start, spoken_at = await self.start_turn(audio)
        await self.finish_turn(start)
        return turn_metrics(self.recorder, start, spoken_at + audio.speech_end_s)

    async def barge_in(self, answer: PromptAudio, interruption: PromptAudio) -> dict[str, Any]:
        """Ask for a long answer, talk over it once it is audible, measure the cut-over."""
        start, _ = await self.start_turn(answer)
        ended = await self.recorder.wait("audio", start, lambda v: v[0] == "SPEECH_ENDED")
        await self.recorder.wait("enqueue", self.recorder.marks.index(ended))
        guard_s = self.settings.barge_in_echo_guard_ms / MS
        await asyncio.sleep(guard_s + BARGE_IN_DELAY_SECONDS)
        if self.state != "speaking":
            raise BenchmarkError(f"answer ended before the interruption ({self.state})")
        onset = self.recorder.position()
        push = self.recorder.add("push", int(self.feed.delivered() + self.feed.pending()))
        self.feed.push(interruption.samples)
        await self.recorder.wait("state", onset, lambda v: v == ("barge-in", "listening"))
        await self.finish_turn(self.recorder.position())
        return barge_in_metrics(self.recorder, onset, push, interruption.speech_start_s)


class BenchmarkError(RuntimeError):
    """A run did not go as scripted; it is reported and skipped, not measured."""


class ApplicationStoppedError(JarvisError):
    """The assistant under test exited; the scenario cannot continue."""


def replay_engine(paths: Any, feed: Any, wake_word: bool, min_speech_ms: int | None) -> Any:
    """The real ``jarvis_audio.AudioEngine`` with ``feed`` in place of the microphone."""
    module = _native.load("jarvis_audio")
    config = module.EngineConfig()
    if min_speech_ms is not None:
        config.pipeline.segmenter.min_speech_ms = min_speech_ms
    models = (paths.melspectrogram, paths.embedding, paths.wake_word) if wake_word else None
    return module.AudioEngine(paths.vad, models, config, replay=feed)


def session_settings(spec: SessionSpec) -> Any:
    """Environment settings with the benchmark's mode, language and a scratch data dir."""
    data_dir = STATE_DIR / f"{spec.mode}-{spec.language}"
    shutil.rmtree(data_dir, ignore_errors=True)
    overrides: dict[str, Any] = {
        "data_dir": data_dir,
        "tts_voice": JARVIS_VOICES[spec.language],
        "stt_language": None if spec.language == "en" else spec.language,
        "overlay_enabled": False,
        "tray_enabled": False,
        "dashboard_enabled": False,
        "proactive_enabled": False,
    }
    if spec.mode == "local":
        overrides |= {"mode": RuntimeMode.LOCAL, "llm_provider": "local"}
    else:
        overrides |= {"mode": RuntimeMode.CLOUD, "llm_provider": "ollama"}
    return Settings().model_copy(update=overrides)


def providers_of(settings: Any) -> dict[str, str]:
    """Human-readable provider names for the JSON report."""
    if settings.mode.value == "local":
        return {
            "stt": f"whisper.cpp {settings.local_stt_model.name}",
            "llm": f"llama.cpp {settings.local_llm_model.name}",
            "tts": f"Piper {settings.local_tts_model.name}",
        }
    return {
        "stt": f"faster-whisper {settings.stt_model} ({settings.stt_device})",
        "llm": f"Ollama {settings.ollama_model}",
        "tts": f"edge-tts {settings.tts_voice}",
    }


def turn_metrics(recorder: Recorder, start: int, last_word_at: float) -> dict[str, Any]:
    """Stage durations of the turn that began at mark ``start``."""
    metrics: dict[str, Any] = {}
    for mark in recorder.marks[start:]:
        if mark.kind == "sample" and mark.value[0] in SAMPLE_STAGES:
            metrics.setdefault(SAMPLE_STAGES[mark.value[0]], round(mark.value[1], 1))
    transcript = recorder.find("transcript", start)
    metrics["transcript"] = None if transcript is None else transcript.value
    metrics["tools"] = [mark.value for mark in recorder.marks[start:] if mark.kind == "tool"]
    speech_end = recorder.find("audio", start, lambda v: v[0] == "SPEECH_ENDED")
    if speech_end is None:
        return metrics
    first_audio = recorder.find("enqueue", recorder.marks.index(speech_end))
    if first_audio is not None:
        metrics[SPEECH_END_TO_AUDIO.key] = round((first_audio.at - speech_end.at) * MS, 1)
        metrics[LAST_WORD_TO_AUDIO.key] = round((first_audio.at - last_word_at) * MS, 1)
    return metrics


def barge_in_metrics(
    recorder: Recorder, onset: int, push: Mark, speech_start_s: float
) -> dict[str, Any]:
    """Speech onset -> native stop, and native stop -> LISTENING.

    ``push`` holds the feed position where the interruption was queued, at
    the wall-clock moment it was queued. The engine's ``SPEECH_STARTED``
    sample position (the native stop happens in the same worker iteration)
    maps back to wall-clock time through that anchor, because the replay
    source delivers exactly one second of samples per second.
    """
    started = recorder.find("audio", onset, lambda v: v[0] == "SPEECH_STARTED")
    listening = recorder.find("state", onset, lambda v: v == ("barge-in", "listening"))
    if started is None or listening is None:
        raise BenchmarkError("barge-in was not observed")
    stop_offset_s = (started.value[1] - push.value) / SAMPLE_RATE
    stopped_at = push.at + stop_offset_s
    return {
        BARGE_IN_TO_STOP.key: round((stop_offset_s - speech_start_s) * MS, 1),
        BARGE_IN_TO_LISTENING.key: round((listening.at - stopped_at) * MS, 1),
    }


# --- driver ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Options:
    """Command-line options."""

    modes: tuple[str, ...]
    languages: tuple[str, ...]
    runs: int
    sink: str
    barge_in_runs: int
    output: Path
    log_level: str


def parse_args(argv: Sequence[str] | None = None) -> Options:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--modes", default="cloud,local", help="comma-separated: cloud, local")
    parser.add_argument("--languages", default="en,sv", help="comma-separated: en, sv")
    parser.add_argument("--runs", type=int, default=3, help="measured runs per prompt")
    parser.add_argument("--barge-in-runs", type=int, default=3, help="0 skips barge-in")
    parser.add_argument(
        "--sink",
        choices=("null", "device"),
        default="null",
        help="null: measure up to the device hand-off in silence (default); device: real speakers",
    )
    parser.add_argument("--output", type=Path, default=RESULTS_FILE)
    parser.add_argument("--log-level", default="WARNING", help="Jarvis log level (stderr)")
    args = parser.parse_args(argv)
    return Options(
        modes=tuple(args.modes.split(",")),
        languages=tuple(args.languages.split(",")),
        runs=args.runs,
        sink=args.sink,
        barge_in_runs=args.barge_in_runs,
        output=args.output,
        log_level=args.log_level,
    )


def scenarios_for(options: Options) -> Iterator[SessionSpec]:
    """Mode x language, minus Swedish in local mode (no Swedish Piper voice is downloaded)."""
    for mode in options.modes:
        for language in options.languages:
            if mode == "local" and language != "en":
                continue
            yield SessionSpec(mode, language, options.sink)


async def measure_prompt(
    session: Session, result: ScenarioResult, *, audio: PromptAudio, runs: int
) -> None:
    """One warm-up turn, then ``runs`` measured turns of ``audio``."""
    for index in range(runs + 1):
        try:
            metrics = await session.turn(audio)
        except (BenchmarkError, TimeoutError) as error:
            result.errors.append(f"{audio.prompt.key} run {index}: {error!r}")
            continue
        if index > 0:
            result.runs.append({"prompt": audio.prompt.key, **metrics})
        print(f"  {audio.prompt.key} #{index}: {summary_line(metrics)}", flush=True)


async def measure_barge_in(session: Session, result: ScenarioResult, *, runs: int) -> None:
    """``runs`` scripted interruptions of a long answer."""
    answer = await load_prompt(BARGE_IN_ANSWER)
    interruption = await load_prompt(BARGE_IN_INTERRUPTION)
    for index in range(runs):
        try:
            metrics = await session.barge_in(answer, interruption)
        except (BenchmarkError, TimeoutError) as error:
            result.errors.append(f"barge-in run {index}: {error!r}")
            continue
        result.runs.append({"prompt": BARGE_IN_ANSWER.key, **metrics})
        print(f"  barge-in #{index}: {summary_line(metrics)}", flush=True)


def summary_line(metrics: dict[str, Any]) -> str:
    """Progress output for one run."""
    timings = {k: v for k, v in metrics.items() if isinstance(v, float)}
    text = ", ".join(f"{key}={value:.0f}" for key, value in timings.items())
    transcript = metrics.get("transcript")
    return text + (f' heard "{transcript}"' if transcript else "")


Job = Callable[[Session, ScenarioResult], Awaitable[None]]


async def scenario_jobs(spec: SessionSpec, options: Options) -> list[Job]:
    """One job per prompt, plus the barge-in runs for English."""
    jobs: list[Job] = []
    for prompt in PROMPTS[spec.language]:
        audio = await load_prompt(prompt)
        jobs.append(partial(measure_prompt, audio=audio, runs=options.runs))
    if spec.language == "en" and options.barge_in_runs > 0:
        jobs.append(partial(measure_barge_in, runs=options.barge_in_runs))
    return jobs


async def run_scenario(spec: SessionSpec, options: Options) -> ScenarioResult:
    """Measure every job of one mode and language, each in a running application.

    If the application under test crashes, the crash is recorded (it is a
    finding, not noise) and a fresh application is built for the next job,
    so one failure does not hide every later measurement.
    """
    print(f"\n== {spec.mode} / {spec.language}", flush=True)
    result = ScenarioResult(spec.mode, spec.language, providers_of(session_settings(spec)))
    pending = await scenario_jobs(spec, options)
    while pending:
        async with Session(spec) as session:
            while pending:
                try:
                    await pending.pop(0)(session, result)
                except ApplicationStoppedError as error:
                    report_crash(error, result)
                    break
    return result


def report_crash(error: ApplicationStoppedError, result: ScenarioResult) -> None:
    """Keep the crash, with its traceback, in the JSON and on the console."""
    cause = error.__cause__
    trace = "" if cause is None else "".join(traceback.format_exception(cause))
    result.errors.append(f"{error}\n{trace}".strip())
    print(f"  {error}\n{trace}", flush=True)


async def run(options: Options) -> list[ScenarioResult]:
    """Every scenario in turn; one that cannot start is reported, not fatal."""
    results: list[ScenarioResult] = []
    for spec in scenarios_for(options):
        try:
            results.append(await run_scenario(spec, options))
        except JarvisError as error:
            print(f"  skipped {spec.mode} / {spec.language}: {error}", flush=True)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark, print the table and write the JSON report."""
    options = parse_args(argv)
    configure_logging(options.log_level)
    results = asyncio.run(run(options))
    if not results:
        print("nothing was measured")
        return 1
    hardware = hardware_info()
    config = {"runs_per_prompt": options.runs, "warm_up_runs": 1, "sink": options.sink}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    document = report_json(results, hardware=hardware, config=config)
    options.output.write_text(json.dumps(document, indent=2, ensure_ascii=False), "utf-8")
    print("\n" + render_table(results) + "\n\n" + caption(hardware, options.runs, options.sink))
    print(f"\nJSON written to {options.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# Design notes:
# - The microphone is replaced below the engine (a native ReplaySource), not
#   above it: replaying into AudioEngine keeps the real VAD, segmenter
#   hangover, pre-roll and native barge-in in the measurement, which is
#   where most of the felt latency and all of the barge-in behaviour live.
# - Turns start with the StubPlatform hotkey rather than a synthesised
#   "hey Jarvis": the wake-word model still runs on every frame, but a
#   detection miss on synthetic speech would measure the detector's recall,
#   not the pipeline's latency.
# - Importing this module needs no native build: jarvis loads jarvis_audio
#   lazily, so the tests of the pure helpers run in the Linux CI lane too.
# - Barge-in onset -> stop is measured on the engine's sample clock: the
#   native stop runs in the same worker iteration as SPEECH_STARTED, so the
#   event's sample position is the stop's position. It includes the
#   segmenter's minimum speech duration (250 ms by default), deliberately:
#   that is the time the user talks over Jarvis before it falls silent.
