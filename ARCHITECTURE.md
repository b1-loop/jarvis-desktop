# Architecture

This document is the detailed companion to the overview in `README.md`. It
describes the code as it is and is updated in the same pull request as any
change it describes; drift is a bug. The reasons behind each decision live
in the [ADRs](docs/adr/README.md).

## 1. Guiding principle

**Python is the conductor, C++ is the muscle.** Python owns control flow,
policy and integration; C++ owns anything with a real-time or latency budget.
The boundary is pybind11 and it is crossed as rarely as possible: the native
side batches audio into frames, the Python side batches decisions into tool
calls. Native threads never call into Python and never take the GIL; Python
polls (audio events, hotkeys, generated tokens) and pushes (overlay state,
playback chunks).

## 2. Repository layout

```
jarvis-desktop/
  native/                   C++20, CMake presets, pybind11
    common/                 version, errors, cancellation token
    audio_engine/           capture, ring buffer, VAD, wake word, playback, barge-in, MP3 decoding, replay source
    os_integration/         Win32: hotkeys, windows, processes, text injection, notifications, system controls, screenshots, clipboard
    inference/              whisper.cpp and llama.cpp wrappers
    overlay/                Dear ImGui holographic HUD on Direct3D 11 with GPU bloom (ADR-0023)
    bindings/               pybind11 modules: jarvis_audio, jarvis_os, jarvis_inference, jarvis_overlay
    tests/                  Catch2, one executable
    third_party/            whisper.cpp, llama.cpp (git submodules)
  src/jarvis/
    __main__.py             CLI (typer): --mode, --text-only, --no-wakeword, --background
    config.py               pydantic-settings, the only place settings are read from the environment
    user_settings.py        data/user_settings.json layered over .env (the HUD's settings page)
    _native/                loader for the compiled modules (the .pyd files are git-ignored)
    app/                    windowless desktop app: background session, single instance, shortcuts, icon, Ollama autostart
    core/                   composition root (bootstrap, voice_bootstrap, providers, ui_wiring,
                            proactive_wiring), event bus, state machine, orchestrator,
                            voice I/O, triggers, barge-in, latency probes, overlay bridge
    audio/                  asyncio facade over jarvis_audio, capture watchdog, model manifest
    os_integration/         Platform interface, WindowsPlatform over jarvis_os, StubPlatform
    stt/ tts/ llm/          interfaces + providers
    inference/              loader and model manifest for jarvis_inference
    agent/                  loop, tool base, registry, confirmation, chaining, tool selection, persona
    tools/                  63 tools in 60 plugin modules (helpers start with "_")
    memory/                 SQLite store, vector stores, memory manager, user profile
    proactive/              scheduler, triggers, YAML routines
    integrations/           email, calendar, chat, music, smart home, web, documents; Stub* providers
      connect/              one-time account sign-in from a terminal (connect_accounts.py)
    vision/                 screenshot -> downscale -> vision-capable LLM
    ui/                     overlay facade, tray icon, dashboard, view model
    utils/                  errors, logging, timing, retry, sandbox, downloads, colour (OKLCH)
  tests/                    mirrors src/, plus tests/scripts/ for the scripts
  scripts/                  build_native, download_models, record_test_audio, benchmark_latency,
                            benchmark_local, compare_models, demo_ui, demo_overlay,
                            install_shortcuts, make_icon, connect_accounts, record_demo.md
  routines/                 YAML routines (morning briefing)
  docs/                     adr/, tools.md (tool catalogue), media/
```

## 3. Pipeline and data flow

```
wake word / hotkey → VAD → capture → STT → agent (LLM + tools) → TTS → playback
```

Each stage is a class behind an abstract base or Protocol in its package's
`base.py`. Stages never call each other; the orchestrator moves data between
them and publishes progress on the event bus. Text and voice share one
orchestrator, one agent and one state machine; only the front end differs
(ADR-0014, ADR-0019).

### Composition

`core/bootstrap.py` builds everything from `Settings`, in dependency order:
event bus → SQLite store → memory → LLM → tool registry → platform → tool
services (integrations, background jobs, vision) → front end (terminal or
voice) → agent → orchestrator → proactive scheduler → dashboard and tray.
Every component registers its cleanup on one `AsyncExitStack`, so shutdown
runs in reverse and a failure half-way through closes what was opened.
`__main__` only parses the command line and calls it. The front end is a
builder (`FrontEndBuilder`) so the voice one (`core/voice_bootstrap.py`) can
use the bus, state machine and platform built before it and hand back input,
output, a confirmation asker, an LLM latency probe, barge-in wiring and the
overlay and microphone controls.

### Text path

```
TerminalInput ─line─▶ Orchestrator ─text─▶ Agent ─messages─▶ LLMClient.stream
  (daemon thread)       │  IDLE→LISTENING→THINKING   │  ◀── TextDelta / ToolCallRequested / Done
                        │                            ├─ SentenceSplitter ─sentence─▶ sink
                        │                            └─ ToolChain ─▶ ConfirmationPolicy ─▶ Tool.execute
                        ◀── sentences (THINKING→SPEAKING) ──┘
TerminalOutput ◀─say─┘  SPEAKING→IDLE when the turn ends
```

The orchestrator speaks to two Protocols, `InputSource` and `OutputSink`;
the voice path replaces the terminal with STT and TTS and keeps the rest,
including `cancel_current_turn()` for barge-in. Typed input passes through
LISTENING because the state table has no IDLE → THINKING edge, which keeps
the state sequence identical in both modes. Confirmation questions are read
from the same input source (`TerminalConfirmation`, y/n, unclear answers
count as no). "exit", "quit", "bye" and "goodbye" end the session.

### Voice path

```
              ┌──── AnyTrigger: WakeWordTrigger (engine event) | HotkeyTrigger (Platform) ────┐
VoiceInput ◀──┘  earcon (80 ms tone) → IDLE→LISTENING → AudioEngine.record_utterance
   │                                       (300 ms pre-roll + speech until SPEECH_ENDED)
   │  STT.transcribe (worker thread) → TranscriptReady
   ▼
Orchestrator ─▶ Agent ─▶ FirstTokenProbe(LLMClient) ─▶ sentences ─▶ VoiceOutput.say (queue only)
                                                                    │
         synthesiser task: one TTS.synthesize task per sentence ─▶ │ bounded hand-off (2)
         player task:      Speaker.play(sentence n) ──▶ native Playback ──▶ speakers
                                                                    │ first chunk → BargeInController
                                                                    │ arms engine after echo guard
mic ─▶ native VAD: SpeechStarted ─▶ Playback::stop() (C++, no Python) │
                                     └─▶ VoiceOutput sees the stop ─▶ drop queued sentences,
                                         BargeInController: capture on, cancel turn,
                                         SPEAKING→LISTENING ─▶ VoiceInput skips the trigger
```

`core/voice_bootstrap.py` assembles this front end: the engine with the wake
word if its models load, playback at the TTS sample rate, the mode's STT and
TTS from `core/providers.py` (the STT model is loaded and runs one silent
decode, and a local Piper voice is loaded, before the greeting; the LLM loads
in the background from `core/bootstrap.py`, ADR-0030),
the hotkey, and the overlay bridge. `OutputSink.flush()` is awaited inside the
turn task, so SPEAKING lasts as long as the audio does and a barge-in during
the last sentence still cancels the turn. Echo is handled by the segmenter's
minimum speech duration (`JARVIS_VAD_MIN_SPEECH_MS`, 250 ms) plus an echo
guard that arms barge-in only 400 ms after playback starts; there is no
acoustic echo cancellation (ADR-0019). Confirmation questions are spoken
(`VoiceConfirmation`) with barge-in suspended, and the answer is heard
without a trigger.

### Providers

| Stage | Interface | Default (`cloud` mode) | Alternatives | `local` mode |
| --- | --- | --- | --- | --- |
| STT | `stt.base.STT` | `FasterWhisperSTT` (CTranslate2, off the loop; CPU int8 fallback when CUDA libraries are missing) | | `WhisperCppSTT` |
| TTS | `tts.base.TTS` | `EdgeTTS` (MP3 decoded natively in batches) | `PiperTTS` (`JARVIS_TTS_PROVIDER=piper`: the local voice, no network round trip) | `PiperTTS` |
| LLM | `llm.base.LLMClient` | `OllamaClient` (`qwen2.5:7b`) | `AnthropicClient` (`JARVIS_LLM_PROVIDER=anthropic`; streaming, tool use, vision) | `LocalLlamaClient` |
| Embeddings (tool selection) | `llm.embeddings.TextEmbedder` | `OllamaEmbedder` (`bge-m3`) | none: word overlap only (`JARVIS_TOOL_EMBEDDING_MODEL=none`) | same as `cloud` when Ollama runs, else word overlap |

`core/providers.py` picks the stack from `Settings.mode` and
`llm.factory.create_llm_client` the LLM from `JARVIS_LLM_PROVIDER`; a
missing prerequisite raises `ConfigError` naming the fix (extra, native
build, model download) instead of silently switching providers. Providers
receive their external dependency through a `Protocol`, so unit tests run
on fakes and never touch the network or the native build. Streaming is
sentence by sentence: `llm.streaming.SentenceSplitter` cuts text deltas into
sentences, each sentence goes to `TTS.synthesize`, which yields PCM chunks
while the rest downloads. Retries share one policy in `utils.retry`
(ADR-0005).

**LOCAL mode.** `WhisperCppSTT` and `LocalLlamaClient` sit on
`jarvis_inference` through Protocols; every native call runs in
`asyncio.to_thread`, one hop per generated token, and cancelling the
consumer cancels the native generation. The llama.cpp wrapper has no tool
channel, so `llm.json_tool_protocol` renders the offered tool schemas into
the system prompt, classifies the streamed reply as prose or one
`{"tool", "arguments"}` object, validates it against the schema and falls
back to text. `PiperTTS` yields one chunk per sentence from the `piper-tts`
package. In LOCAL mode an `anthropic` provider setting is replaced by
llama.cpp (logged), so `--mode local` never sends a request off the machine
(ADR-0013).

## 4. State machine

`IDLE → LISTENING → THINKING → SPEAKING → IDLE`. `core/state_machine.py`
holds the current `AssistantState` and an explicit transition table:

| From | To | Meaning |
| --- | --- | --- |
| IDLE | LISTENING | wake word or hotkey (typed input passes through too) |
| LISTENING | THINKING | end of speech, transcript ready |
| LISTENING | IDLE | silence or nothing usable heard |
| THINKING | SPEAKING | answer (or tool summary) ready |
| THINKING | IDLE | nothing to say |
| SPEAKING | IDLE | playback finished |
| SPEAKING | LISTENING | **barge-in**: the user talked over Jarvis |

Anything else raises `InvalidTransitionError`. Every transition is published
as a `StateChanged` event on the `EventBus` (`core/events.py`), next to
`WakeWordHeard`, `TranscriptReady`, `AssistantSaid`, `ToolCallStarted` /
`ToolCallFinished`, `ConfirmationRequested` / `ConfirmationResolved` and
`LatencySample`. That is how the overlay, tray icon, dashboard and logs stay
in sync without the orchestrator knowing about them. The bus is in-process
and single-loop; each subscriber has a bounded queue that drops the oldest
event (counted and logged) rather than ever blocking the publisher, because
the publisher is the latency-critical voice path. Synchronous callbacks
(`subscribe_callback`) exist for listeners that must see an event at the
moment it is published, such as the latency benchmark.

## 5. Threading model

| Thread | Owner | Work | Shares |
| --- | --- | --- | --- |
| Event loop | Python asyncio | orchestrator, agent, tools, bus; polls audio events every 10 ms and hotkeys every 50 ms; enqueues TTS chunks | engine mutex, playback mutex (brief) |
| Executor threads | Python (`asyncio.to_thread`) | STT, Piper, native OS calls, screenshots and imaging, document parsing, ChromaDB, token waits for llama.cpp | none beyond their call |
| SQLite connection | aiosqlite | one thread per connection runs every query | its request queue |
| Terminal input | `TerminalInput` (text mode) | blocking `input()`; lines go back with `call_soon_threadsafe` | none |
| Capture callback | miniaudio (C++) | push device frames into the SPSC ring buffer and bump the received-sample counter, nothing else | ring buffer (producer), atomic counter |
| Device notifications | miniaudio (C++, WASAPI notification client) | forward stopped / rerouted / interruption to the engine as events | engine mutex (brief) |
| Audio worker | `AudioEngine` (C++) | pop frames, run `AudioPipeline` (Silero VAD, openWakeWord, segmentation, pre-roll), append events, trigger barge-in | ring buffer (consumer), engine mutex |
| Playback callback | miniaudio (C++) | drain the `Playback` chunk queue under `try_lock`; silence when contended or empty | playback mutex (never blocks) |
| Hotkey loop | `HotkeyManager` (C++) | `RegisterHotKey` is thread-affine; registers and pumps `WM_HOTKEY` into a queue Python polls | event queue mutex |
| Notifier loop | `BalloonNotifier` (C++) | owns the hidden window and tray icon `Shell_NotifyIcon` needs | task queue mutex |
| Generation | `GenerationStream` (C++) | llama.cpp decode loop; pushes pieces into a queue that Python pulls with the GIL released | stream mutex, cancellation token |
| ggml / CTranslate2 pools | whisper.cpp, llama.cpp, faster-whisper | intra-op worker threads inside one inference call | owned by the library |
| Overlay UI thread | `OverlayWindow` (C++) | Win32 message loop, one model snapshot per frame, Dear ImGui + D3D11 with the bloom pass, DPI changes | overlay model mutex (read) |
| Tray thread | `TrayIcon` (Python, pystray) | notification-area icon and menu; clicks go back with `call_soon_threadsafe` | pre-rendered icon images (read-only) |
| Dashboard thread | `Dashboard` (Python, tkinter) | owns the Tk interpreter; drains a queue of immutable snapshots every 100 ms | snapshot queue |
| Replay source | `ReplaySource` (C++, benchmark only) | delivers queued samples in 10 ms frames at the real-time rate instead of a microphone | replay feed mutex |

Rules that keep this safe:

- The capture callback never allocates, locks or runs inference; overflow
  drops samples and counts them (`SamplesDropped` event) instead of stalling.
- Native threads never call into Python and never take the GIL. Python
  polls; `start()`/`stop()`, transcription and token waits release the GIL.
- `AudioPipeline` is single-threaded by construction. The engine wraps every
  call to it in one mutex; the Python side only takes that mutex for
  microsecond-scale operations (poll, toggle listening, take audio). ONNX
  Runtime sessions use one intra-op thread so the models never compete with
  the audio worker.
- The overlay never calls back into Python: the event loop pushes state,
  levels and text through setters that take the model mutex for
  nanoseconds, and the UI thread draws a snapshot (ADR-0011). What the user
  does in the interactive HUD is queued natively and polled by the bridge
  every 30 ms (ADR-0024).
- The tray and dashboard threads never touch the event bus or the view
  model: the loop pushes immutable snapshots (dashboard) or swaps a
  pre-rendered image (tray), and user actions come back only through
  `call_soon_threadsafe` (ADR-0015).
- A capture that stops delivering is reopened from the event loop, never
  from a device thread: the `CaptureWatchdog` task reads the lock-free
  `samplesReceived()` counter every 250 ms and calls `restartCapture()`
  through `asyncio.to_thread` after 2.5 s without samples or after a
  `DeviceStopped` event. A lifecycle mutex serialises the restart with
  `start()`/`stop()`, and the engine mutex is never held while the device
  opens or closes (ADR-0027).
- Barge-in is `Playback::stop()` called by the worker on `SpeechStarted`
  when a barge-in target is set, so it takes effect on the next device
  period without waiting for the event loop. Python learns about it
  afterwards via `Playback::stopCount()`.

### Audio engine data flow

```
mic or ReplaySource ──callback──▶ ring buffer ──worker──▶ AudioPipeline
                                          ├─ 512-sample frames ─▶ Silero VAD ─▶ SpeechSegmenter ─▶ SpeechStarted/Ended
                                          ├─ 1280-sample chunks ▶ openWakeWord (mel → embedding → classifier) ─▶ WakeWordDetected
                                          ├─ pre-roll (300 ms) ──▶ seeds capture when listening starts
                                          └─ captured audio ─────▶ takeCaptured() ─▶ STT
```

Detection parameters (thresholds, hysteresis, minimum speech 250 ms as
configured by Jarvis, 700 ms silence hangover, cooldown, pre-roll) live in
`PipelineConfig` and are exposed to Python unchanged. ADR-0003 records why
this layer is C++.

Silero v5 scores each 512-sample frame together with the last 64 samples
of the previous one, as the reference implementation does; without that
context the probability flickered on short, quiet requests and the
segmenter never saw enough speech frames after the wake word. While the
segmenter counts towards `SpeechStarted`, a frame between the end and start
thresholds pauses the count and only a frame below the end threshold
restarts it, the same hysteresis that keeps speech alive once it has begun.

### OS integration

Tools talk to the operating system only through
`jarvis.os_integration.Platform` (an ABC with async methods and plain
dataclasses). `WindowsPlatform` implements it over `jarvis_os` and runs every
native call in `asyncio.to_thread`; `StubPlatform` records calls for
text-only mode, tests and the latency benchmark. On the C++ side each
capability is one header/source pair behind a small interface
(`WindowManager`, `ProcessControl`, `SystemControls`, `Screenshotter`,
`TextInjector`, `Notifier`, `Clipboard`) with no Win32 types in public
headers. Windows, processes, volume, brightness, screenshots and the
clipboard run on the calling executor thread with per-call COM
initialisation; hotkeys and notifications own the two message-loop threads
in the table above (ADR-0004).

## 6. Agent and tools

A tool is one module in `jarvis/tools/` defining a `Tool` subclass: `name`,
`description`, a pydantic `Input` model, a `category`, a risk declaration
(`requires_confirmation` or a per-call `needs_confirmation`) and
`async run()`. `agent/registry.py` discovers every plugin with `pkgutil`,
records broken ones as `DiscoveryFailure`s instead of crashing, and produces
the `{name, description, input_schema}` definitions from the pydantic
models. The catalogue with confirmation flags and services is
[docs/tools.md](docs/tools.md).

Tools receive services only through `ToolContext`:

| Field | What it is |
| --- | --- |
| `settings`, `bus`, `clock` | configuration, the event bus, a monotonic clock |
| `ask_confirmation` | the front end's yes/no question (terminal or spoken) |
| `platform` | the `Platform` interface |
| `memory` | notes, todos, reminders and facts (`MemoryManager` satisfies the Protocol) |
| `integrations` | `IntegrationServices`: email, calendar, chat, music, smart home, web search, YouTube transcripts (ADR-0018) |
| `vision` | the `ScreenReader`, or `None` when the provider cannot see images |
| `jobs` | `BackgroundJobs`, the tracker of processes started by dev tools (ADR-0017) |

A tool returns a `ToolResult` with `spoken` (for TTS), `detail` (for the LLM
and dashboard) and `data` (JSON). Invalid arguments and tool failures become
`ok=False` results, never exceptions in the agent loop.

**Confirmation.** `agent/confirmation.py` is the only place that decides
whether to ask the user: the tool's declaration, a list of always-confirm
categories and the `JARVIS_CONFIRM_DESTRUCTIVE` trusted-mode flag. Shell
commands additionally pass through `utils/sandbox.py` (argv only, denylist,
allowlist, working root, scrubbed environment, timeout, output cap);
dev-mode tools run fixed command lines through the same sandbox, and
`start_dev_server` uses `spawn_sandboxed` and `BackgroundJobs` (ADR-0006,
ADR-0017).

**The loop.** `agent/loop.py` runs one request: a system prompt from
`agent/persona.py` (name, tone, language from the TTS voice, spoken-style
rules, the rule that live data comes from a tool call, and the current local
date and time from the injected clock, ADR-0029) plus the
`MemoryManager.recall` context, the user's words and the tool definitions chosen by `agent/tool_selection.py` go to
`LLMClient.stream`; text deltas become sentences for the sink as they
complete; requested tools run and their results go back to the model until
it stops or `JARVIS_MAX_TOOL_ROUNDS` is reached. A turn with tools always
ends with a spoken summary (asked for once if the model is silent,
otherwise built from the tools' own `spoken` results). LLM failures are
spoken as an apology with the provider's remedy and, for cloud providers,
the LOCAL/Ollama hint. The turn is recorded in memory with its tool calls.

**Tool selection.** For Ollama and llama.cpp each round offers at most
`JARVIS_MAX_TOOLS_PER_TURN` (default 12) tools, plus a core set
(`JARVIS_CORE_TOOLS`) and every tool already used this turn; Claude gets all
tools. The ranking blends word overlap (English and Swedish keywords) with
cosine similarity from a multilingual Ollama embedding model
(`JARVIS_TOOL_EMBEDDING_MODEL`, default `bge-m3`): tool vectors are embedded
once and cached under `data/cache/`, each request is embedded within 150 ms,
and anything slower or failing falls back to word overlap
(`agent/tool_embeddings.py`). When every tool call of a turn
failed or was declined, the loop ends with the tools' own `spoken` sentences
instead of the model's final text (ADR-0020).

**Chaining.** `agent/chaining.py` makes the chain robust: one `ToolChain` per
request with a tool-call budget, bounded parallel execution of the calls in
one model message (confirmation prompts serialised behind a lock),
de-duplication of identical calls, and a `TurnReport` (tools used, rounds,
durations, failures, limits hit) for the dashboard (ADR-0014).

## 7. Memory

`memory/MemoryManager` is the only memory API the agent sees. Behind it:

| Store | Holds | Role |
| --- | --- | --- |
| `SqliteStore` (aiosqlite, WAL, versioned migrations) | conversations, messages, notes, reminders, todos, facts | source of truth; FTS5 search with `LIKE` fallback |
| `VectorStore` Protocol: `ChromaVectorStore` / `InMemoryVectorStore` | embeddings of turns and notes | recall by meaning; rebuildable from SQLite |

`UserProfile` is a view over `profile.*` facts. `recall()` returns profile,
facts, recent turns and semantic hits trimmed to a character budget.
Answers that came from tools are never part of it: those exchanges are left
out of the recent turns and their answers are cut off semantic hits, so live
data is always fetched again (ADR-0029). Recalled turns are framed as history,
not instructions; the profile's name travels separately (`user_name`) and the
persona states it as the form of address above them. A conversation's
summary is written when the session ends. All data stays under
`Settings.data_dir` (ADR-0008).

## 8. Proactive behaviour and routines

`proactive/scheduler.py` is one asyncio scheduler with an injectable clock.
`proactive/triggers.py` holds polled triggers (`UpcomingMeetingTrigger`,
`DueReminderTrigger`, `LowBatteryTrigger`, `LongRunningCommandTrigger`)
that check their source every `JARVIS_PROACTIVE_POLL_SECONDS`.
`proactive/routines.py` loads YAML routines from `routines/` (time or phrase
triggers, tool steps, spoken intro and outro, per-step timeouts).
`core/proactive_wiring.py` connects them to the running assistant: sources
from memory, the calendar tools, the background jobs and a psutil battery
probe; an `IdleAnnouncer` that speaks (or prints) a notification only once
the assistant is idle and also raises a system notification; and
`RoutineAwareTurns`, which answers a phrase routine inside a turn instead
of the agent. Routine tool steps go through the same confirmation policy as
agent tool calls (ADR-0007).

## 9. Screen understanding

`vision/ScreenReader` turns "what does this error mean?" into one request:
capture through `Platform` (primary monitor, whole desktop, or the focused
window), a `Redactor` hook (no-op today), downscale to
`vision_max_long_edge` and re-encode PNG, then JPEG, then smaller until the
base64 fits `vision_max_image_bytes` (`vision/imaging.py`, in a worker
thread), and one user message with the image and the question to a
vision-capable `LLMClient`. `vision/factory.py` picks that client from the
configured provider (Claude with `vision_model`, Ollama only with
`ollama_vision_model`, none for llama.cpp). Tools reach it through
`ToolContext.vision`; `describe_screen`, `explain_screen_error` and
`ask_about_screen` say aloud when a screenshot went to a cloud provider.
Captures happen only inside a tool call (ADR-0016).

## 10. Local inference

`native/inference` wraps whisper.cpp and llama.cpp (ADR-0010).
Transcription is one blocking call with the GIL released. Generation runs on
a native thread that fills a queue; Python pulls pieces from a
`GenerationStream` and can cancel it, so the LLM path follows the same
no-GIL-on-native-threads rule as the audio engine. The Python providers over
this module are described under Providers above.

## 11. Overlay, tray and dashboard

`core/overlay_bridge.py` subscribes to the bus and pushes state, the
transcript, the spoken reply and a level pump (the microphone level while
listening, the playback level while speaking) into the native overlay
through `ui/overlay.py`; `StubOverlay` stands in when the module or a
desktop is missing. `core/ui_wiring.py` builds the pystray tray icon (state
colour; menu: show dashboard, toggle overlay, mute microphone, quit) and the
tkinter dashboard (live state, transcript, tool calls with durations, stage
latency statistics, memory inspector) for both front ends; both run on their
own threads and receive immutable snapshots from `ui/view_model.py`
(ADR-0011, ADR-0015).

The HUD is interactive (ADR-0024). Its shapes are hit-tested natively
(transparent pixels stay click-through), it can be dragged, typed into,
expanded to a conversation view and used to change settings. The bridge
drains its UI events and dispatches them to `core/hud_wiring.py`;
`core/combined_input.py` merges typed lines and voice turns into the
orchestrator's one input (typing while Jarvis speaks barges in),
`core/reply_output.py` implements "speak replies", and
`ui/hud_settings.py` validates, saves (`data_dir/user_settings.json`, read
back at start by `jarvis/user_settings.py` between the environment and
`.env`) and applies settings live or after a restart.

It can also be minimized to a small glowing orb (ADR-0024, update): the
docking, the morph, auto-minimize and full-screen hiding are pure native
modules (`overlay/orb.hpp`, `overlay/minimize_policy.hpp`); Python
remembers the user's choice (`core/hud_wiring.py`), offers it in the tray,
and registers the `hud_toggle_hotkey` (`core/hud_hotkey.py`) through a
`core/hotkey_router.py` shared with the voice hotkey so neither consumer
steals the other's presses.

Closing the HUD hides it until it is needed (ADR-0025): the overlay
bridge shows it again on the next `WakeWordHeard` (wake word or voice
hotkey) with the ordinary wake ripple, keeping an orb an orb, and a
launch while Jarvis runs replays the full boot (`Overlay.replay_boot`).

### Desktop app

`scripts/install_shortcuts.py` writes desktop, Start-menu and (optionally)
Startup shortcuts that run `uvw run --project <checkout> jarvis
--background`, plus "Jarvis HUD" desktop and Start-menu shortcuts that add
`--show-hud`. `--background` (or a process without stdout) selects the
windowless session in `jarvis.app.background`: logs go to a rotating
JSON-lines file (`data/logs/jarvis.log`), an OS file lock
(`data/jarvis.lock`) keeps one instance and a second launch asks the first,
over a loopback socket served on the event loop with a random token, to
come forward (`HandoverRequest`): `HudActions.bring_forward` shows the
HUD, restores it from the orb and replays the boot through the native
`replayBoot`, and the plain shortcut also shows the dashboard (ADR-0025); Ollama is started hidden when it is the LLM and not
answering; voice mode runs with the dashboard hidden and the tray icon as
the only surface. A failure is reported as a notification plus a tray
tooltip naming the log (ADR-0021).

## 12. Error handling and degradation

One exception hierarchy rooted at `jarvis.utils.errors.JarvisError`
(`ConfigError`, `NativeUnavailableError`, `LLMError`, `TranscriptionError`,
`SynthesisError`, `PlatformError` and its subclasses, `MemoryStoreError`,
`VisionError`). Errors are logged with context through structlog (secrets
redacted by a processor) and never swallowed silently. The degradation
ladder, as implemented:

| Failure | What happens | Where |
| --- | --- | --- |
| `jarvis_audio` not built, VAD model missing, no microphone or output device | voice mode says why and starts text mode | `voice_bootstrap.run_voice_mode` |
| Wake-word models missing or broken | warning once, hotkey only, greeting says so | `VoiceFrontEndBuilder._create_engine` |
| Microphone stops delivering (sleep/resume, system stall, device invalidated or unplugged) | reopened on the current default device with backoff (1 s up to 30 s); one notification if three attempts fail, another when it is back | `audio/watchdog.py` |
| Hotkey taken by another app | wake word only; fatal only if neither exists | `VoiceFrontEndBuilder._trigger` |
| TTS fails at runtime | one notice, answers printed | `VoiceOutput._report_failure` |
| Nothing usable heard | "didn't catch that", back to IDLE | `VoiceInput._next_request` |
| CUDA libraries missing for faster-whisper | CPU int8, one warning naming the library | `stt/faster_whisper.py` |
| LLM unreachable or failing | spoken apology with the remedy; for cloud providers the LOCAL/Ollama hint | `agent/loop.py` |
| Selected mode's prerequisite missing (extra, build, model) | `ConfigError` naming the fix | `core/providers.py` |
| `jarvis_os` not built | `StubPlatform`, logged | `bootstrap.build_platform` |
| ChromaDB missing or unopenable | in-memory vector store for the session; SQLite failure is fatal | `bootstrap.build_vector_store` |
| No integration credentials | `Stub*` providers, or "that integration isn't configured" | `integrations/factory.py` |
| Account needs consent outside a terminal (voice, windowless) | `ConsentRequiredError` spoken as "run `connect_accounts.py <service>`", one notification per account; no browser, no device code | `integrations/consent.py` |
| A tool plugin fails to import | `DiscoveryFailure`, logged, others load | `agent/registry.py` |
| Vision unavailable for the provider | the screen tools say so | `bootstrap.build_vision` |
| psutil missing | battery trigger skipped with a warning | `bootstrap.build_battery_source` |
| Overlay, tray or dashboard cannot start | stubs; the session continues | `ui/` factories |
| Windowless app cannot start voice | notification and tray tooltip with the reason and the log path; Quit exits | `app/background.py` |
| Ollama not running (windowless app) | started hidden, waited for up to `JARVIS_OLLAMA_START_TIMEOUT_SECONDS` | `app/ollama_launcher.py` |

## 13. Native build

`scripts/build_native.py` runs the CMake presets (`windows-release`,
`windows-debug`, `ci-windows` with sccache) and the build writes the `.pyd`
files and `onnxruntime.dll` into `src/jarvis/_native/`; the Python package
itself is pure Python, so a C++ toolchain is only needed for voice, OS,
local inference and overlay features (ADR-0002). Options:
`JARVIS_BUILD_TESTS`, `JARVIS_BUILD_INFERENCE` (off builds a stub
`jarvis_inference` that only reports its status) and
`JARVIS_BUILD_OVERLAY`. Third-party code comes in pinned: miniaudio and the
ONNX Runtime binary by URL hash through FetchContent, Catch2 and Dear ImGui
by tag, whisper.cpp and llama.cpp as submodules. Warnings are errors
(`/W4 /WX`) for our code only; `--tidy` runs clang-tidy, which the CI
workflow shards four ways; the same commands run locally (see
`CONTRIBUTING.md`).

## 14. Latency budget

Each voice stage is published as a `LatencySample` (`core/voice_latency.py`)
for the dashboard and the benchmark: `voice.wake_to_listening`,
`voice.speech_end_to_transcript`, `voice.transcript_to_first_token` (via
`FirstTokenProbe`, which wraps the `LLMClient`) and
`voice.first_sentence_to_audio`. `scripts/benchmark_latency.py` adds what no
single stage sees: end of speech -> first audio handed to the device, last
word -> first audio, and the barge-in cut-over, by replaying synthesised
prompts through a `ReplaySource` in place of the microphone so the native
VAD, segmenter and barge-in run unchanged. The targets are for cloud mode
with GPU speech recognition; the measured column is the benchmark run of
2026-10-22 on the development machine (median / p90 of 9 English turns,
3 barge-ins; faster-whisper `small` on the CPU because cuBLAS is not
installed, Ollama `qwen2.5:7b` on the GPU, edge-tts, null playback sink).
The README has the full table including Swedish and local mode, and the
Swedish GPU run after the latency work (ADR-0030), in which the benchmark
also splits the stages into their parts (STT decode, Ollama model load and
prompt evaluation, TTS first chunk) from the providers' log events.

| Stage | Where | Target | Measured, cloud mode |
| --- | --- | --- | --- |
| Trigger -> listening (80 ms earcon) | `voice.wake_to_listening` | < 150 ms | 89 / 94 ms |
| VAD hangover after the last word | segmenter `silenceEndMs` | 700 ms | about 750 ms per run (last word -> first audio minus end of speech -> first audio: hangover, 32 ms VAD frames, 10 ms poll) |
| End of speech -> transcript | `voice.speech_end_to_transcript` | < 600 ms (GPU) | 4414 / 4712 ms (CPU, loaded machine; 2.1 s median in an earlier run) |
| Transcript -> first LLM event | `voice.transcript_to_first_token` | < 800 ms | 2848 / 3370 ms |
| First sentence -> first audio | `voice.first_sentence_to_audio` | < 500 ms | 1066 / 1865 ms (edge-tts round trip per sentence) |
| End of speech -> first audio | benchmark | sum of the above | 9193 / 10193 ms |
| Barge-in onset -> playback stopped | native `Playback::stop` on `SpeechStarted` | 250 ms minimum speech, then < 1 device period | 450 / 450 ms (detection; the stop itself is in the same worker iteration) |
| Playback stopped -> LISTENING | Python cleanup | < 50 ms | 30 / 38 ms (23 ms in the step-7 check) |

The gap between target and measurement is almost entirely CPU speech
recognition and the 7B model's time to first token (Ollama, prompt with
memory and up to 12 tool schemas); the pipeline's
own overhead (trigger, sentence pipelining, barge-in cleanup) is within
budget. edge-tts pays a service round trip per sentence (first byte 0.6 s
median, 1.8 s p90), which is why the first-sentence stage varies between
0.6 and 3 s; sentences are therefore synthesised concurrently, and the
local Piper voice (about 0.1 s to its first chunk) is the configurable way
around it. On the GPU, with the model kept loaded (`JARVIS_OLLAMA_KEEP_ALIVE`)
and Ollama reached over IPv4, end of speech -> first audio measured
2.7 / 3.6 s with Piper and 3.8 / 4.5 s with edge-tts (ADR-0030).

## 15. Trade-offs

Collected in `docs/adr/`. The headline ones: C++ for the audio hot path
(0003), interfaces in front of every provider and the OS (0004, 0005),
plugin tools with central confirmation (0006), SQLite plus Chroma (0008),
own thin inference wrappers (0010), CLOUD vs LOCAL (0013), one orchestrator
for text and voice (0014), native-first barge-in (0019), per-turn tool
selection for small models (0020) and shortcuts to `uvw` rather than a
frozen executable (0021).
