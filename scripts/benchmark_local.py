"""Measure CLOUD vs LOCAL provider latency on this machine (ADR-0013's table).

Usage::

    uv run python scripts/benchmark_local.py            # all rows whose prerequisites exist
    uv run python scripts/benchmark_local.py --runs 5

Rows:

- STT latency for the same 3 s utterance: faster-whisper ``small`` (CLOUD)
  vs whisper.cpp ``tiny`` (LOCAL).
- LLM time to first token and tokens/s: Ollama (if a server answers at
  ``JARVIS_OLLAMA_BASE_URL``), llama.cpp with the pinned Qwen2.5 0.5B, and
  Claude (only when ``ANTHROPIC_API_KEY`` is set).
- TTS time to first audio for one sentence: edge-tts vs Piper.

Each provider is warmed up once first (model load or connection set-up is
reported in its own column), then measured ``--runs`` times; the median is
printed. A row whose prerequisite is missing is listed as skipped with the
reason instead of failing the run.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import platform
import statistics
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from jarvis.config import RuntimeMode, Settings
from jarvis.core.providers import build_llm, build_stt, build_tts
from jarvis.llm.base import Done, LLMClient, Message, SystemMessage, TextDelta, UserMessage
from jarvis.stt.base import STT
from jarvis.stt.faster_whisper import resample_linear
from jarvis.tts.base import TTS
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import configure_logging

UTTERANCE_SECONDS = 3.0
MS_DISPLAY_LIMIT_S = 10.0
CPU_REGISTRY_KEY = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
STT_RATE = 16_000
UTTERANCE_TEXT = "Hey Jarvis, what is on my calendar for tomorrow morning?"
TTS_SENTENCE = "Your first meeting tomorrow is at nine with the design team."
LLM_PROMPT: list[Message] = [
    SystemMessage("You are Jarvis, a concise voice assistant."),
    UserMessage("In three sentences, explain why local speech recognition protects privacy."),
]


@dataclass
class Row:
    """One measured (or skipped) provider."""

    stage: str
    provider: str
    warm_up_s: float | None = None
    first_s: float | None = None
    rate: float | None = None
    note: str = ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=3, help="measured runs per row (median)")
    return parser.parse_args(argv)


async def timed(action: Callable[[], Awaitable[Any]]) -> float:
    """Seconds ``action`` takes."""
    started = time.perf_counter()
    await action()
    return time.perf_counter() - started


async def median_of(runs: int, action: Callable[[], Awaitable[float]]) -> float:
    """Median of ``runs`` sequential measurements."""
    return statistics.median([await action() for _ in range(runs)])


# --- STT ------------------------------------------------------------------------


async def utterance(settings: Settings) -> tuple[np.ndarray, str]:
    """Exactly 3 s of 16 kHz speech: Piper's voice if available, else a tone."""
    try:
        tts = build_tts(settings.model_copy(update={"mode": RuntimeMode.LOCAL}))
        chunks = [chunk async for chunk in tts.synthesize(UTTERANCE_TEXT)]
    except JarvisError as error:
        tone = np.sin(np.linspace(0, 2 * np.pi * 220 * UTTERANCE_SECONDS, int(STT_RATE * 3)))
        return (0.1 * tone).astype(np.float32), f"220 Hz tone (Piper unavailable: {error})"
    speech = resample_linear(np.concatenate([c.samples for c in chunks]), tts.sample_rate, STT_RATE)
    padded = np.zeros(int(STT_RATE * UTTERANCE_SECONDS), dtype=np.float32)
    padded[: min(speech.size, padded.size)] = speech[: padded.size]
    return padded, f'Piper speaking "{UTTERANCE_TEXT}"'


async def bench_stt(name: str, settings: Settings, audio: np.ndarray, runs: int) -> Row:
    """Load time, then median transcription latency of ``audio``."""
    row = Row("STT (3 s audio)", name)
    try:
        stt: STT = build_stt(settings)
        row.warm_up_s = await timed(lambda: stt.transcribe(audio[:STT_RATE]))
        row.first_s = await median_of(runs, lambda: timed(lambda: stt.transcribe(audio)))
        row.note = f'heard: "{(await stt.transcribe(audio)).text}"'
    except JarvisError as error:
        row.note = f"skipped: {error}"
    return row


# --- LLM ------------------------------------------------------------------------


async def one_turn(client: LLMClient) -> tuple[float, float]:
    """Time to first text delta and output tokens/s after it."""
    started = time.perf_counter()
    first: float | None = None
    async for event in client.stream(LLM_PROMPT):
        if isinstance(event, TextDelta) and first is None:
            first = time.perf_counter()
        if isinstance(event, Done):
            ended = time.perf_counter()
            if first is None:
                raise JarvisError("the model produced no text")
            tokens = max(event.usage.output_tokens - 1, 1)
            return first - started, tokens / max(ended - first, 1e-6)
    raise JarvisError("stream ended without Done")


async def bench_llm(name: str, build: Callable[[], LLMClient], runs: int) -> Row:
    """Warm-up turn (includes model load), then median TTFT and tokens/s."""
    row = Row("LLM", name)
    try:
        client = build()
        row.warm_up_s = await timed(lambda: one_turn(client))
        results = [await one_turn(client) for _ in range(runs)]
        row.first_s = statistics.median(ttft for ttft, _ in results)
        row.rate = statistics.median(rate for _, rate in results)
    except JarvisError as error:
        row.note = f"skipped: {error}"
    return row


async def ollama_model(settings: Settings) -> str | None:
    """The configured Ollama model if the server runs and has it, else the first installed."""
    try:
        async with httpx.AsyncClient(base_url=settings.ollama_base_url, timeout=3.0) as http:
            tags = (await http.get("/api/tags")).json()
    except httpx.HTTPError:
        return None
    names = [model["name"] for model in tags.get("models", [])]
    return settings.ollama_model if settings.ollama_model in names else next(iter(names), None)


async def llm_rows(settings: Settings, runs: int) -> list[Row]:
    """Ollama, llama.cpp and Claude rows, each only if available."""
    rows: list[Row] = []
    model = await ollama_model(settings)
    if model is None:
        rows.append(
            Row("LLM", "Ollama", note=f"skipped: no server/model at {settings.ollama_base_url}")
        )
    else:
        ollama = settings.model_copy(update={"llm_provider": "ollama", "ollama_model": model})
        rows.append(await bench_llm(f"Ollama {model}", lambda: build_llm(ollama), runs))
    local = settings.model_copy(update={"mode": RuntimeMode.LOCAL, "llm_provider": "local"})
    rows.append(
        await bench_llm(f"llama.cpp {local.local_llm_model.name}", lambda: build_llm(local), runs)
    )
    if os.environ.get("ANTHROPIC_API_KEY"):
        rows.append(
            await bench_llm(f"Claude {settings.llm_model}", lambda: build_llm(settings), runs)
        )
    else:
        rows.append(Row("LLM", "Claude", note="skipped: ANTHROPIC_API_KEY not set"))
    return rows


# --- TTS ------------------------------------------------------------------------


async def first_audio(tts: TTS) -> float:
    """Seconds until the first chunk of :data:`TTS_SENTENCE` arrives."""
    started = time.perf_counter()
    stream = tts.synthesize(TTS_SENTENCE)
    await anext(stream)
    elapsed = time.perf_counter() - started
    async for _ in stream:
        pass
    return elapsed


async def bench_tts(name: str, settings: Settings, runs: int) -> Row:
    """Warm-up sentence, then median time to first audio."""
    row = Row("TTS (first audio)", name)
    try:
        tts = build_tts(settings)
        row.warm_up_s = await timed(lambda: first_audio(tts))
        row.first_s = await median_of(runs, lambda: first_audio(tts))
    except JarvisError as error:
        row.note = f"skipped: {error}"
    return row


# --- report ---------------------------------------------------------------------


def hardware() -> str:
    """CPU model, core counts, RAM and OS, for the ADR."""
    cpu = platform.processor()
    if sys.platform == "win32":
        # Imported by name: Windows-only (and only for a readable CPU name).
        winreg = importlib.import_module("winreg")
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, CPU_REGISTRY_KEY)
        cpu = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    try:
        psutil = importlib.import_module("psutil")  # optional: the desktop extra
    except ImportError:
        return f"{cpu}, {platform.platform()}"
    ram_gb = psutil.virtual_memory().total / 2**30
    cores = f"{psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads"
    return f"{cpu} ({cores}), {ram_gb:.0f} GB RAM, {platform.platform()}"


def seconds(value: float | None) -> str:
    """Milliseconds below 10 s, seconds above; a dash when not measured."""
    if value is None:
        return "-"
    return f"{value * 1000:.0f} ms" if value < MS_DISPLAY_LIMIT_S else f"{value:.1f} s"


def render(rows: list[Row], runs: int) -> str:
    """The Markdown table pasted into ADR-0013."""
    lines = [
        f"Hardware: {hardware()}. Median of {runs} runs after one warm-up.",
        "",
        "| Stage | Provider | Warm-up (load) | Latency / TTFT / first audio | Tokens/s | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        rate = f"{row.rate:.1f}" if row.rate is not None else "-"
        lines.append(
            f"| {row.stage} | {row.provider} | {seconds(row.warm_up_s)} | "
            f"{seconds(row.first_s)} | {rate} | {row.note} |"
        )
    return "\n".join(lines)


async def run(runs: int) -> list[Row]:
    """Every row, in table order."""
    settings = Settings()
    audio, source = await utterance(settings)
    print(f"STT input: {source}", file=sys.stderr)
    cloud = settings.model_copy(update={"mode": RuntimeMode.CLOUD})
    # Pinned to the CPU so both STT rows compare like for like; "auto" picks
    # CUDA whenever CTranslate2 sees a device, even without the CUDA runtime.
    cpu_stt = cloud.model_copy(update={"stt_device": "cpu", "stt_compute_type": "int8"})
    local = settings.model_copy(update={"mode": RuntimeMode.LOCAL})
    rows = [
        await bench_stt(
            f"faster-whisper {settings.whisper_model(gpu=False)} (cpu, int8)", cpu_stt, audio, runs
        ),
        await bench_stt(f"whisper.cpp {settings.local_stt_model.name}", local, audio, runs),
    ]
    rows += await llm_rows(settings, runs)
    rows.append(await bench_tts(f"edge-tts {settings.tts_voice}", cloud, runs))
    rows.append(await bench_tts(f"Piper {settings.local_tts_model.stem}", local, runs))
    return rows


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    configure_logging("WARNING")
    print(render(asyncio.run(run(args.runs)), args.runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
