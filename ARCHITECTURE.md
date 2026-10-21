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

## 4. State machine *(step 6)*

`IDLE → LISTENING → THINKING → SPEAKING → IDLE`. Transitions are validated by
`core/state_machine.py` and published as `StateChanged` events. Barge-in is the
transition `SPEAKING → LISTENING` triggered by VAD activity during playback.

## 5. Threading model *(step 3)*

| Thread | Owner | Work |
| --- | --- | --- |
| Audio callback | miniaudio (C++) | copy device frames into the ring buffer, nothing else |
| Audio worker | C++ | VAD and wake-word inference on ring-buffer frames |
| Playback | C++ | feed TTS chunks to the output device, honour cancellation |
| Event loop | Python asyncio | everything else |
| Executor threads | Python | STT and other CPU-bound Python work |

The GIL is released for every native call that can block. Native threads
never call into Python; they push into lock-free queues that the Python side
drains.

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
