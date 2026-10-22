# CLAUDE.md

You are a senior software architect and engineer. We are building "Jarvis", a voice-controlled personal AI assistant for the desktop, as a portfolio project for a developer CV. It will be reviewed by recruiters and senior engineers, so architecture, code quality, tests and documentation matter as much as functionality. Treat every file as if a senior reviewer will read it.

The guiding story of this project: **"Python is the conductor, C++ is the muscle."** All orchestration, agent logic and tools live in Python; every hot path (audio, native OS hooks, local inference, overlay UI) lives in C++ and is exposed to Python via pybind11.

## Target platform

Primary OS: **Windows 11**. No external hardware; the app only needs the computer's microphone and speakers. The OS layer sits behind a common interface (`jarvis.os_integration.Platform`) so macOS and Linux can be added later, but only Windows is implemented now.

Toolchain on the development machine: Python 3.12, uv, CMake 4, Ninja, MSVC from Visual Studio 2026 (`C:\Program Files\Microsoft Visual Studio\18\Insiders`), LLVM 23 for clang-format and clang-tidy.

## Tech stack (non-negotiable)

- Python 3.12 for orchestration: agent loop, tool plugins, memory, API clients, schedulers.
- C++20 for all performance-critical and native code, built with CMake, exposed via pybind11:
  1. Audio engine: mic capture, ring buffer, VAD (Silero VAD via ONNX Runtime), wake-word detection (openWakeWord ONNX models), playback with barge-in support. Audio I/O via miniaudio.
  2. Native OS integration layer: global hotkeys, active window title/app, window management, text injection at cursor, system notifications, volume/brightness, lock screen, screenshot. Win32 API on Windows.
  3. Local inference wrappers: whisper.cpp for offline STT and llama.cpp for offline LLM, built from source as submodules and wrapped in our own thin C++ API.
  4. Status overlay: a small always-on-top window with live waveform, state indicator and transcript, written with Dear ImGui on DirectX 11.
- LLM: behind an `LLMClient` interface with native tool calling and streaming. Default: Ollama with `qwen2.5:7b` (free, local; `OllamaClient`). Optional cloud upgrade: Anthropic Claude via the official SDK (`JARVIS_LLM_PROVIDER=anthropic`). Offline: `LocalLlamaClient` using our llama.cpp wrapper. Vision-capable model calls for screen understanding.
- STT: faster-whisper (default) and whisper.cpp (offline) behind an `STT` interface.
- TTS: edge-tts (default) and Piper (offline) behind a `TTS` interface. Streaming playback so speech starts before the full answer is generated.
- Storage: SQLite for conversation history, notes, reminders and long-term facts; ChromaDB for semantic memory.
- Config: pydantic-settings, `.env` (never committed), committed `.env.example`. API keys only from environment variables.
- Tooling: uv for Python deps, ruff (lint + format), mypy --strict, pytest + pytest-asyncio, pytest-cov, pre-commit hooks, clang-format, clang-tidy, GitHub Actions CI that runs lint, type-check, Python tests and the C++ build + tests on every push.

## Architecture (follow exactly)

Pipeline: wake word → VAD → audio capture → STT → agent (LLM + tools) → TTS → playback. Each stage is an independent module behind an abstract base class / Protocol. Stages communicate via asyncio queues and an event bus. A central state machine: IDLE → LISTENING → THINKING → SPEAKING, with transitions published as events so the overlay, logs and tray icon stay in sync. Barge-in: the user can talk over Jarvis and it stops speaking immediately.

Two runtime modes selectable in config: CLOUD (faster-whisper + edge-tts, with the LLM from `JARVIS_LLM_PROVIDER`: Ollama `qwen2.5:7b` by default, Claude optional) and LOCAL (llama.cpp + whisper.cpp + Piper, fully offline). Trade-offs (latency, quality, privacy, cost) are documented in ADR-0013; measured end-to-end latency is in the README (`scripts/benchmark_latency.py`).

Native extensions are built separately from the Python package (`scripts/build_native.py` → `src/jarvis/_native/*.pyd`) so text-only development and the pure-Python CI jobs never need a C++ compiler. See `docs/adr/0002-decoupled-native-build.md`.

Project layout: see `ARCHITECTURE.md`. The short version:

```
native/        C++20: common, audio_engine, os_integration, inference, overlay, bindings, tests
src/jarvis/    Python: core, audio, os_integration, stt, tts, llm, agent, tools, memory,
               proactive, integrations, vision, ui, utils
tests/         mirrors src/, unit + integration, mocked audio/LLM/network
scripts/       download models, record test audio, benchmark latency, build native
docs/adr/      Architecture Decision Records
```

## Features, in phases

### Phase 1 – Core
- Wake word ("Hey Jarvis", configurable) plus global hotkey fallback
- Speak and answer with voice, streaming TTS, barge-in
- Text-only mode in the terminal for development
- Persistent memory between sessions: who the user is, what they work on, what was discussed
- Computer control: open/close apps, switch windows, volume, brightness, screenshot, lock screen, type text at cursor
- Run shell commands and scripts with allowlist/denylist and spoken confirmation for anything destructive
- Status overlay + tray icon

### Phase 2 – Rich integrations
- Email (Gmail + Outlook): list unread, summarize, search, draft replies approved by voice, send after confirmation
- Calendar (Google + Outlook): agenda, free slots, create events, remind 10 minutes before
- Web: search, read an article aloud, summarize a YouTube video from its transcript
- Files: find documents, summarize PDF/docx/txt/md, "open the last thing I worked on"
- Screen understanding: screenshot → vision model
- Dev mode: git status, run tests, explain a stack trace, open repo in VS Code, start dev server, push to GitHub
- Messaging: Slack/Discord/Teams – read what matters, reply after confirmation
- Music: Spotify play/pause/search/queue
- Notes and todo: dictate, list, search, tick off
- System monitoring: CPU, RAM, battery, top resource consumers
- Smart home: Home Assistant / Philips Hue provider (interface + stub if no hub)

### Phase 3 – Wow factor
- Proactive behaviour: upcoming meeting, important email, low battery, long-running command finished
- Routines: "good morning" briefing; configurable routines in YAML
- Multi-step task chaining: one request, several tools, one spoken summary
- Document production: Word / Google Docs / Markdown
- Personality: configurable name, voice, tone, persona; remembers habits and preferences
- Fully offline LOCAL mode
- Dashboard window: live state, transcript, tool calls, timings, memory inspector
- Latency benchmark script and a results table in README

## Tool and agent rules

- Every capability is a plugin: one Python file in `tools/` with name, description, pydantic input schema and async `run()`. Dropping in a file adds a capability; no central registration.
- Tools that modify the system, delete, send, spend money or touch external accounts set `requires_confirmation=True`. The agent enforces this centrally and asks the user for spoken confirmation before executing.
- The agent must handle free-form phrasing, chain multiple tools in one turn, and always end with a short spoken summary of what it did.
- Never execute LLM output as code. Validate all tool inputs with pydantic. Sandbox shell commands.

## Code quality rules (ALWAYS follow, never skip)

- Clean Code: functions under 30 lines, single responsibility, descriptive names, no magic numbers, no dead code, no commented-out code.
- SOLID and dependency injection. No module imports a concrete implementation from another layer, only its interface.
- Full type hints everywhere. `mypy --strict` must pass.
- Google-style docstrings on every public module, class and function. Explain WHY, not only WHAT.
- Inline comments for non-trivial logic only; never comment the obvious.
- Every file with a non-obvious decision ends with a short "Design notes" comment block.
- C++: RAII, no raw owning pointers, const-correctness, clear ownership, Doxygen on all public APIs, `-Wall -Wextra -Werror` (`/W4 /WX` on MSVC), clang-format + clang-tidy clean, unit tests for every component.
- Errors: custom exception hierarchy, never swallow exceptions, structured logging with context, graceful degradation (TTS fails → print; wake word fails → hotkey; cloud down → offer LOCAL mode).
- Async-first; never block the event loop with I/O or CPU work (use executors / native threads).
- Tests before moving on: unit tests for every module, integration tests with mocked audio/LLM/network, target >80% coverage.
- Security: no secrets in code or logs, `.env` ignored by git, least-privilege OAuth scopes.
- Conventional Commits. Commit after every completed step.
- Keep README, ARCHITECTURE.md and ADRs updated as you go; documentation drift is a bug.

## Git workflow

- Remote: `github.com/b1-loop/jarvis-desktop`.
- Every step of work happens on its own branch (`feat/...`, `fix/...`, `docs/...`, `chore/...`), is pushed, and lands on `main` through a pull request. Nothing is committed directly to `main` after the initial scaffold.
- When several branches touch the same files (ADR index, README, ARCHITECTURE, the composition root), they land through an integration branch (`integration/wave-N`): merge them there, resolve conflicts once (renumber colliding ADRs, keep every ADR row), run the full suite on the combined code, then open one PR to `main`. Parallel branches each writing "ADR-0008" is how this rule was learned.
- Run before every commit: `uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -m "not hardware"`, plus the native build and tests when C++ changed (`uv run python scripts/build_native.py`, `--tidy` for clang-tidy). Never commit `.env`, `data/`, `models/` or `.claude/`.
- After a change to `.pre-commit-config.yaml` or `uv.lock`, rerun `uv sync --all-extras --group dev` and `uv run pre-commit install --hook-type pre-commit --hook-type commit-msg`; ruff and mypy run from `uv.lock` in hooks, terminal and CI alike.

## Environment quirks (for future agents)

- Put uv, Python 3.12 and LLVM on `PATH` explicitly in every shell; never call a bare `python` that may resolve to another interpreter.
- Long native builds (whisper.cpp, llama.cpp) exceed a 10-minute foreground limit: run `scripts/build_native.py` in the background. `--skip-tests` builds without running Catch2.
- Ollama serves `qwen2.5:7b` at `http://localhost:11434`. The development machine has an NVIDIA GPU (8 GB, shared with Ollama) but no CUDA toolkit: `uv sync --all-extras` installs the `gpu` extra (cuBLAS/cuDNN wheels) so faster-whisper runs on CUDA; without it it falls back to the CPU. Set `JARVIS_STT_DEVICE=cpu` for CPU benchmarks.
- The workstation may be locked while agents run: desktop-dependent native and Python tests skip themselves (session lock probe), clipboard and window calls fail on a locked desktop, and CI runners may have no audio endpoint.
- The repository is public and GitHub Actions runs `.github/workflows/ci.yml` on every push and pull request. Still run every check locally before pushing (CI is a second gate, not the first), and wait for CI to pass before merging.
- Someone may be sitting at the computer: never play audio through the real speakers from automated runs. `scripts/benchmark_latency.py` uses a silent null sink by default (`--sink device` only when asked), and `hardware`-marked tests are run by hand only.
- Set `PYTHONIOENCODING=utf-8` when redirecting Jarvis output to a file (the Windows console code page cannot encode every character the tools print).
- Shared scratch files from parallel agents need a unique prefix; a stray `wave.py` in a shared scratch directory once shadowed the standard library for scripts run from there (run helper scripts with `python -I`).

## How to work

1. Plan the step. 2. Scaffold. 3. Audio engine. 4. OS layer. 5. STT/TTS/LLM cloud providers. 6. Agent core and text-only loop. 7. Voice pipeline with barge-in. 8. Phase 1 and 2 tools. 9. Memory. 10. Proactive. 11. Vision. 12. LOCAL mode. 13. Overlay and dashboard. 14. Docs, ADRs, benchmark.

After each step: run ruff, mypy, pytest and the C++ build + tests. Fix everything before continuing. If something is ambiguous, state the assumption in a comment and keep going; only stop if truly blocked. Never leave a TODO without an explanation of what is needed. Never silently fake an implementation; stubs must be obvious in the class name (e.g. `StubEmailProvider`) and in the docs.
