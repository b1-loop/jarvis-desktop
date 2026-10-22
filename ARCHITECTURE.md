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
    third_party/          whisper.cpp, llama.cpp (git submodules, pinned to release tags)
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
    ui/                   overlay facade, tray icon, dashboard
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

### Text path *(step 6)*

```
TerminalInput ─line─▶ Orchestrator ─text─▶ Agent ─messages─▶ LLMClient.stream
  (daemon thread)       │  IDLE→LISTENING→THINKING   │  ◀── TextDelta / ToolCall / Done
                        │                            ├─ SentenceSplitter ─sentence─▶ sink
                        │                            └─ ToolChain ─▶ ConfirmationPolicy ─▶ Tool.execute
                        ◀── sentences (THINKING→SPEAKING) ──┘
TerminalOutput ◀─say─┘  SPEAKING→IDLE when the turn ends
```

`core/bootstrap.py` builds everything from `Settings` (event bus, memory,
LLM, tool registry, platform, integrations, agent, orchestrator) and owns
cleanup in one `AsyncExitStack`; `__main__` only calls it. The orchestrator
speaks to two Protocols, `InputSource` and `OutputSink`, so the voice path
(step 7) replaces the terminal with STT and TTS and keeps the rest,
including `cancel_current_turn()` for barge-in. Typed input passes through
LISTENING because the state table has no IDLE → THINKING edge, which keeps
the state sequence identical in both modes. Confirmation questions are read
from the same input source (`TerminalConfirmation`, y/n, unclear answers
count as no). See ADR-0014.

### Providers *(step 5)*

| Stage | Interface | CLOUD | Ollama option | LOCAL *(step 12)* |
| --- | --- | --- | --- | --- |
| STT | `stt.base.STT` | `FasterWhisperSTT` (CTranslate2, off-loop) | same | `WhisperCppSTT` |
| TTS | `tts.base.TTS` | `EdgeTTS` (MP3 decoded natively in batches) | same | `PiperTTS` |
| LLM | `llm.base.LLMClient` | `AnthropicClient` (streaming, tool use, vision) | `OllamaClient` | `LocalLlamaClient` |

`llm.factory.create_llm_client` picks the LLM by `JARVIS_LLM_PROVIDER`.
Providers receive their external dependency through a `Protocol`, so unit
tests run on fakes and never touch the network or the native build.
Streaming is sentence-by-sentence: `llm.streaming.SentenceSplitter` cuts
text deltas into sentences, each sentence goes to `TTS.synthesize`, which
yields PCM chunks while the rest downloads. Retries share one policy in
`utils.retry`. See ADR-0005.

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
| Overlay UI thread | `OverlayWindow` (C++) | Win32 message loop, one model snapshot per frame, Dear ImGui + D3D11 rendering, DPI changes | overlay model mutex (read) |
| Tray thread | `TrayIcon` (Python, pystray) | Win32 message loop for the notification-area icon; menu clicks go back to the loop with `call_soon_threadsafe` | pre-rendered icon images (read-only) |
| Dashboard thread | `Dashboard` (Python, tkinter) | owns the Tk interpreter and every widget; drains a `queue.Queue` of immutable snapshots every 100 ms; the refresh button hops back with `call_soon_threadsafe` | snapshot queue |
| Generation | `GenerationStream` (C++) | llama.cpp decode loop; pushes text pieces into a queue that Python pulls from with the GIL released | stream mutex, cancellation token |

Rules that keep this safe:

- The capture callback never allocates, locks or runs inference; overflow
  drops samples and counts them (`SamplesDropped` event) instead of stalling.
- Native threads never call into Python and never take the GIL. Python
  polls; `start()`/`stop()` release the GIL while they touch devices.
- `AudioPipeline` is single-threaded by construction. The engine wraps every
  call to it in one mutex; the Python side only takes that mutex for
  microsecond-scale operations (poll, toggle listening, take audio).
- The overlay never calls back into Python either: the event loop pushes
  state, levels and text through setters that take the model mutex for
  nanoseconds, and the UI thread draws a snapshot (ADR-0011).
- The tray and dashboard threads never touch the event bus or the view
  model: the loop pushes immutable snapshots (dashboard) or swaps a
  pre-rendered image (tray), and user actions come back only through
  `call_soon_threadsafe` (ADR-0015).
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

### OS integration

Tools talk to the operating system only through
`jarvis.os_integration.Platform` (an ABC with async methods and plain
dataclasses). `WindowsPlatform` implements it over the `jarvis_os` module
and runs every native call in `asyncio.to_thread`; `StubPlatform` records
calls for text-only mode and tests. On the C++ side each capability is one
header/source pair behind a small interface (`WindowManager`,
`ProcessControl`, `SystemControls`, `Screenshotter`, `TextInjector`,
`Notifier`, `Clipboard`) with no Win32 types in public headers.

Two more native threads join the table above, both created lazily and
released by `Platform.close()`:

| Thread | Owner | Work | Shares |
| --- | --- | --- | --- |
| Hotkey loop | `HotkeyManager` (C++) | `RegisterHotKey` is thread-affine; this thread registers and pumps `WM_HOTKEY` into a queue Python polls | event queue mutex |
| Notifier loop | `BalloonNotifier` (C++) | owns the hidden window and tray icon that `Shell_NotifyIcon` requires to outlive any Python executor thread | task queue mutex |

Everything else (windows, processes, volume, brightness, screenshot,
clipboard) runs on the calling executor thread with per-call COM
initialisation. ADR-0004 records the interface split and the notification
and brightness choices.

## 6. Agent and tools *(step 6)*

A tool is one module in `jarvis/tools/` defining a `Tool` subclass: `name`,
`description`, a pydantic `Input` model, a `category`, a risk declaration
(`requires_confirmation` or a per-call `needs_confirmation`) and `async run()`.
`agent/registry.py` discovers every plugin with `pkgutil`, records broken ones
as `DiscoveryFailure`s instead of crashing, and produces the Anthropic-style
`{name, description, input_schema}` definitions from the pydantic models.

Tools receive services only through `ToolContext` (settings, event bus,
clock, confirmation callback, the `Platform` interface, a notes/todos/
reminders Protocol that `MemoryManager` satisfies, and an optional
background-job tracker) and return
a `ToolResult` with `spoken` (for TTS), `detail` (for the LLM and dashboard)
and `data` (JSON). Invalid arguments and tool failures become `ok=False`
results, never exceptions in the agent loop.

The computer-control, file, dev-mode, notes and system tools are listed with
their confirmation flags in [docs/tools.md](docs/tools.md). Dev-mode tools
run fixed command lines through the sandbox; `start_dev_server` uses
`spawn_sandboxed` and `tools/_jobs.BackgroundJobs`, which also feeds the
proactive `LongRunningCommandTrigger` (ADR-0017).

`agent/confirmation.py` is the only place that decides whether to ask the
user: the tool's declaration, a list of always-confirm categories and the
`JARVIS_CONFIRM_DESTRUCTIVE` trusted-mode flag. Shell commands additionally
pass through `utils/sandbox.py` (argv only, denylist, allowlist, working
root, scrubbed environment, timeout, output cap). See ADR-0006.

`agent/loop.py` runs one request: a system prompt from `agent/persona.py`
(name, tone, language from the TTS voice, spoken-style rules) plus the
`MemoryManager.recall` context, the user's words and the registry's tool
definitions go to `LLMClient.stream`; text deltas become sentences for the
sink as they complete; requested tools run and their results go back to the
model until it stops or `JARVIS_MAX_TOOL_ROUNDS` is reached. A turn with
tools always ends with a spoken summary (asked for once if the model is
silent, otherwise built from the tools' own `spoken` results). LLM failures
are spoken as an apology with the provider's remedy and, for cloud
providers, the LOCAL/Ollama hint. The turn is recorded in memory with its
tool calls.

`agent/chaining.py` makes the chain robust: one `ToolChain` per request
with a tool-call budget, bounded parallel execution of the calls in one
model message (confirmation prompts serialised behind a lock),
de-duplication of identical calls, and a `TurnReport` (tools used, rounds,
durations, failures, limits hit) for the dashboard. See ADR-0014.

## 7. Memory *(step 9)*

`memory/MemoryManager` is the only memory API the agent sees. Behind it:

| Store | Holds | Role |
| --- | --- | --- |
| `SqliteStore` (aiosqlite, WAL, versioned migrations) | conversations, messages, notes, reminders, todos, facts | source of truth; FTS5 search with `LIKE` fallback |
| `VectorStore` Protocol: `ChromaVectorStore` / `InMemoryVectorStore` | embeddings of turns and notes | recall by meaning; rebuildable from SQLite |

`UserProfile` is a view over `profile.*` facts. `recall()` returns profile +
facts + recent turns + semantic hits trimmed to a character budget. All data
stays under `Settings.data_dir`. See ADR-0008.

## 7b. Vision *(step 11)*

`vision/ScreenReader` turns "what does this error mean?" into one request:
capture through `Platform` (primary monitor, whole desktop, or only the
focused window), a `Redactor` hook (no-op today), downscale to
`vision_max_long_edge` and re-encode PNG, then JPEG, then smaller until the
base64 fits `vision_max_image_bytes` (`vision/imaging.py`, in a worker
thread), and one user message with the image and the question to a
vision-capable `LLMClient`. `vision/factory.py` picks that client from the
configured provider (Claude with `vision_model`, Ollama only with
`ollama_vision_model`, none for llama.cpp). Tools reach it through
`ToolContext.vision`; `describe_screen`, `explain_screen_error` and
`ask_about_screen` share `tools/_screen.py` and say aloud when a screenshot
went to a cloud provider. Captures happen only inside a tool call. See
ADR-0016.

## 7a. Local inference *(step 12)*

`native/inference` wraps whisper.cpp and llama.cpp (ADR-0010). Transcription
is one blocking call with the GIL released. Generation runs on a native
thread that fills a queue; Python pulls pieces from a `GenerationStream`
and can cancel it, so the LLM path follows the same no-GIL-on-native-threads
rule as the audio engine. The Python `STT` / `LLMClient` providers over this
module are the second half of step 12.

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
