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
`utils.retry`. See ADR-0008.

## 4. State machine *(step 6)*

`IDLE → LISTENING → THINKING → SPEAKING → IDLE`. Transitions are validated by
`core/state_machine.py` and published as `StateChanged` events. Barge-in is the
transition `SPEAKING → LISTENING` triggered by VAD activity during playback.

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

## 6. Agent and tools *(step 6)*

Tool discovery, pydantic validation, confirmation policy and the LLM tool-call
protocol. Covered by ADR-0005 and ADR-0006.

## 7. Memory *(step 9)*

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
