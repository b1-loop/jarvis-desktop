# 0003. C++ for the audio hot path

Date: 2026-10-21
Status: Accepted

## Context

The assistant must listen continuously. Every 32 ms a frame has to pass
through voice activity detection, every 80 ms a chunk through wake-word
detection, and when the user talks over Jarvis, playback must fall silent
within one device period (around 10 ms) or the interruption feels laggy.

The device callback that delivers microphone samples runs on a real-time
thread owned by the OS audio stack. Anything slow or blocking in it (a lock
held by another thread, a memory allocation, a garbage-collection pause, the
GIL) produces audible glitches or dropped audio.

Python can drive audio devices through `sounddevice` or `pyaudio` and run the
same ONNX models through `onnxruntime`, so the choice is not forced by
capability. It is forced by latency predictability and by the project's
guiding story: Python conducts, C++ does the work with a time budget.

## Decision

The audio engine is C++20, exposed to Python through one pybind11 module:

- **miniaudio** opens the default capture and playback devices. It is a
  single header, pinned by hash, and covers WASAPI, CoreAudio and ALSA, so
  the engine has no platform-specific code of its own.
- The capture callback only pushes into a lock-free **SPSC ring buffer**;
  overflow drops samples and counts them rather than ever blocking.
- A **worker thread** pops frames and runs `AudioPipeline`, which hosts
  **Silero VAD** and **openWakeWord** through ONNX Runtime, assembles speech
  segments with hysteresis, keeps a pre-roll buffer and emits events.
- **Playback** keeps a chunk queue that the device thread drains under a
  `try_lock`; `stop()` clears it, so barge-in is a single call that takes
  effect on the next period. The engine issues that call itself when speech
  starts, with no round trip through Python.
- The Python side **polls** for events and captured audio. No native thread
  ever acquires the GIL or calls into Python, which keeps the real-time
  threads independent of the interpreter.
- `AudioPipeline` has no threads, so the whole decision logic is unit-tested
  deterministically with scripted fake detectors; the threaded `AudioEngine`
  is tested with a fake source that feeds frames by hand.

## Alternatives considered

- **Pure Python (`sounddevice` + `onnxruntime`)** – the device callback would
  run Python under the GIL alongside the LLM client, TTS decoding and the
  event loop; wake-word latency becomes jittery and barge-in needs a trip
  through asyncio. Fine for a demo, not for something that should feel
  instant.
- **Rust (`cpal`)** – equally capable; rejected only because the project's
  brief and portfolio story are C++20 and pybind11.
- **PortAudio instead of miniaudio** – mature, but a separate binary
  dependency to build and ship; miniaudio is one header.
- **Porcupine for wake words** – excellent accuracy, but a commercial licence
  and no custom phrases without their console. openWakeWord ships a trained
  "hey Jarvis" model under Apache-2.0 and custom phrases are a config change.
- **WebRTC VAD** – energy-based and noticeably worse with keyboard noise and
  music; Silero costs under a millisecond per frame on a CPU.

## Consequences

- Latency is predictable: detection runs on a dedicated thread with
  sub-millisecond inference, and barge-in is bounded by the device period.
- Voice mode needs the native build; text-only mode does not (ADR-0002).
- `onnxruntime.dll` must sit next to the `.pyd`; the build copies it there.
- The ONNX Runtime download in CMake is currently the Windows x64 release;
  other platforms need their URL added when their OS layer lands.
- Threading rules are documented in `ARCHITECTURE.md` and enforced by
  the code shape: the only object shared between threads is the ring buffer,
  plus one mutex around the pipeline for the rare Python calls.
