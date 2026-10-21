# Architecture

This document is the detailed companion to the overview in `README.md`. It is
updated in the same pull request as any change it describes; drift is a bug.

Sections marked *(step N)* are filled in when that step lands.

## 1. Guiding principle

**Python is the conductor, C++ is the muscle.** Python owns control flow,
policy and integration; C++ owns anything with a real-time or latency budget.
The boundary is pybind11 and it is crossed as rarely as possible: the native
side batches audio into frames, the Python side batches decisions into tool
calls.

## 2. Repository layout

```
jarvis/
  native/                 C++20, CMake, pybind11
    common/               version, cancellation token, shared helpers
    audio_engine/         capture, ring buffer, VAD, wake word, playback, barge-in
    os_integration/       Win32: hotkeys, windows, text injection, notifications, system controls
    inference/            whisper.cpp and llama.cpp wrappers
    overlay/              Dear ImGui status window
    bindings/             pybind11 modules: jarvis_audio, jarvis_os, jarvis_inference, jarvis_overlay
    tests/                Catch2
    third_party/          whisper.cpp, llama.cpp (git submodules, added in step 12)
  src/jarvis/
    __main__.py           CLI entry point and composition root
    config.py             pydantic-settings, the only reader of environment variables
    _native/              shim that loads the compiled modules (git-ignored .pyd files)
    core/                 event bus, state machine, orchestrator, interruption
    audio/                asyncio facade over jarvis_audio
    os_integration/       Platform interface and Windows backend over jarvis_os
    stt/ tts/ llm/        interfaces + providers
    agent/                loop, tool base, registry, confirmation, chaining
    tools/                one plugin per file, auto-discovered
    memory/               SQLite, Chroma, memory manager, user profile
    proactive/            scheduler, triggers, routines
    integrations/         email, calendar, music, chat, smart home providers
    vision/               screenshot to vision model
    ui/                   tray icon, dashboard
    utils/                errors, logging, timing, retry, sandbox
  tests/                  mirrors src/
  scripts/                build_native, download_models, record_test_audio, benchmark_latency
  docs/adr/               Architecture Decision Records
```

## 3. Pipeline and data flow *(step 7)*

```
wake word → VAD → capture → STT → agent (LLM + tools) → TTS → playback
```

Each stage is a class behind an abstract base in its package's `base.py`.
Stages never call each other; the orchestrator moves data between them over
`asyncio.Queue`s and publishes progress on the event bus.

## 4. State machine

`IDLE → LISTENING → THINKING → SPEAKING → IDLE`. `core/state_machine.py`
holds the current `AssistantState` and an explicit transition table:

| From | To | Meaning |
| --- | --- | --- |
| IDLE | LISTENING | wake word or hotkey |
| LISTENING | THINKING | end of speech, transcript ready |
| LISTENING | IDLE | silence, nothing to do |
| THINKING | SPEAKING | answer (or tool summary) ready |
| THINKING | IDLE | nothing to say |
| SPEAKING | IDLE | playback finished |
| SPEAKING | LISTENING | barge-in: the user talked over Jarvis |

Anything else raises `InvalidTransitionError`. Every transition is published
as a `StateChanged` event on the `EventBus` (`core/events.py`), which is how
the overlay, tray icon, dashboard and logs stay in sync without the
orchestrator knowing about them. The bus is in-process and single-loop;
each subscriber has a bounded queue that drops the oldest event (counted and
logged) rather than ever blocking the publisher, because the publisher is
the latency-critical voice path.

## 5. Threading model

| Thread | Owner | Work | Shares |
| --- | --- | --- | --- |
| Capture callback | miniaudio (C++) | push device frames into the SPSC ring buffer, nothing else | ring buffer (producer) |
| Audio worker | `AudioEngine` (C++) | pop frames, run `AudioPipeline` (Silero VAD, openWakeWord, segmentation, pre-roll), append events, trigger barge-in | ring buffer (consumer), engine mutex |
| Playback callback | miniaudio (C++) | drain the `Playback` chunk queue under `try_lock`; silence when contended or empty | playback mutex (never blocks) |
| Event loop | Python asyncio | poll events every 10 ms, orchestrate STT/LLM/TTS, enqueue TTS chunks | engine mutex, playback mutex (brief) |
| Executor threads | Python | STT and other CPU-bound Python work | none |

Rules that keep this safe:

- The capture callback never allocates, locks or runs inference; overflow
  drops samples and counts them (`SamplesDropped` event) instead of stalling.
- Native threads never call into Python and never take the GIL. Python
  polls; `start()`/`stop()` release the GIL while they touch devices.
- `AudioPipeline` is single-threaded by construction. The engine wraps every
  call to it in one mutex; the Python side only takes that mutex for
  microsecond-scale operations (poll, toggle listening, take audio).
- Barge-in is `Playback::stop()` called by the worker on `SpeechStarted`
  when a barge-in target is set, so it takes effect on the next device period
  without waiting for the event loop. Python learns about it afterwards via
  `Playback::stopCount()`.

### Audio engine data flow

```
mic ──callback──▶ ring buffer ──worker──▶ AudioPipeline
                                          ├─ 512-sample frames ─▶ Silero VAD ─▶ SpeechSegmenter ─▶ SpeechStarted/Ended
                                          ├─ 1280-sample chunks ▶ openWakeWord (mel → embedding → classifier) ─▶ WakeWordDetected
                                          ├─ pre-roll (300 ms) ──▶ seeds capture when listening starts
                                          └─ captured audio ─────▶ takeCaptured() ─▶ STT
```

Detection parameters (thresholds, hysteresis, cooldown, pre-roll) live in
`PipelineConfig` and are exposed to Python unchanged. ADR-0003 records why
this layer is C++.

## 6. Agent and tools *(agent loop: step 6, pending)*

A tool is one module in `jarvis/tools/` defining a `Tool` subclass: `name`,
`description`, a pydantic `Input` model, a `category`, a risk declaration
(`requires_confirmation` or a per-call `needs_confirmation`) and `async run()`.
`agent/registry.py` discovers every plugin with `pkgutil`, records broken ones
as `DiscoveryFailure`s instead of crashing, and produces the Anthropic-style
`{name, description, input_schema}` definitions from the pydantic models.

Tools receive services only through `ToolContext` (settings, event bus,
clock, confirmation callback, OS and memory facades as Protocols) and return
a `ToolResult` with `spoken` (for TTS), `detail` (for the LLM and dashboard)
and `data` (JSON). Invalid arguments and tool failures become `ok=False`
results, never exceptions in the agent loop.

`agent/confirmation.py` is the only place that decides whether to ask the
user: the tool's declaration, a list of always-confirm categories and the
`JARVIS_CONFIRM_DESTRUCTIVE` trusted-mode flag. Shell commands additionally
pass through `utils/sandbox.py` (argv only, denylist, allowlist, working
root, scrubbed environment, timeout, output cap). The agent loop that turns
a transcript into tool calls and a spoken summary (`agent/loop.py`) depends
on the LLM interface and lands with it. See ADR-0008.

## 7. Memory *(step 9)*

`memory/MemoryManager` is the only memory API the agent sees. Behind it:

| Store | Holds | Role |
| --- | --- | --- |
| `SqliteStore` (aiosqlite, WAL, versioned migrations) | conversations, messages, notes, reminders, todos, facts | source of truth; FTS5 search with `LIKE` fallback |
| `VectorStore` Protocol: `ChromaVectorStore` / `InMemoryVectorStore` | embeddings of turns and notes | recall by meaning; rebuildable from SQLite |

`UserProfile` is a view over `profile.*` facts. `recall()` returns profile +
facts + recent turns + semantic hits trimmed to a character budget. All data
stays under `Settings.data_dir`. See ADR-0008.

## 8. Native build *(step 2)*

See ADR-0002. `scripts/build_native.py` runs the CMake presets and copies the
resulting `.pyd` files into `src/jarvis/_native/`. The Python package itself is
pure Python, so a C++ toolchain is only needed for voice and OS features.

## 9. Error handling and degradation *(step 6)*

A single exception hierarchy rooted at `jarvis.utils.errors.JarvisError`.
Degradation ladder: TTS fails → print the answer; wake word fails → global
hotkey; cloud unreachable → offer LOCAL mode.

## 10. Trade-offs

Collected in `docs/adr/`. The headline ones are: C++ for the audio hot path,
interfaces everywhere, central confirmation policy, plugin tools, and CLOUD vs
LOCAL mode.
