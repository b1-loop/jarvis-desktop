# audio_engine

Microphone capture, lock-free ring buffer, Silero VAD, openWakeWord and
playback with barge-in. Exposed to Python as `jarvis_audio`.

## Components

| Header | Class | Role |
| --- | --- | --- |
| `ring_buffer.hpp` | `SpscRingBuffer` | lock-free hand-off from the device callback to the worker; counts drops |
| `audio_source.hpp` / `microphone.hpp` | `AudioSource`, `MiniaudioMicrophone` | 16 kHz mono float32 capture |
| `replay_source.hpp` | `ReplayFeed`, `ReplaySource` | virtual microphone: replays queued samples (silence in between) at the real-time rate, for `scripts/benchmark_latency.py` |
| `audio_sink.hpp` / `speaker.hpp` | `AudioSink`, `MiniaudioSpeaker` | pull-based playback device |
| `onnx_session.hpp` | `OnnxSession` | minimal RAII wrapper over ONNX Runtime with name-based tensor access |
| `vad.hpp` / `silero_vad.hpp` | `VoiceActivityDetector`, `SileroVad`, `SpeechSegmenter` | speech probability per 32 ms frame, hysteresis into start/end |
| `wake_word.hpp` / `open_wake_word.hpp` | `WakeWordDetector`, `OpenWakeWord` | three-stage streaming detector, 80 ms chunks |
| `playback.hpp` | `Playback` | chunk queue drained under `try_lock`; `stop()` is barge-in |
| `pipeline.hpp` | `AudioPipeline` | synchronous core: frame assembly, detection, pre-roll, capture, events |
| `engine.hpp` | `AudioEngine` | threads, ring buffer, event queue, barge-in wiring |
| `mp3_decoder.hpp` | `decodeMp3`, `toMono` | decodes a complete MP3 buffer (edge-tts speech) to float32 PCM via dr_mp3 |

Interfaces (`AudioSource`, `AudioSink`, `VoiceActivityDetector`,
`WakeWordDetector`) exist so the pipeline and engine are tested with fakes;
only the miniaudio and ONNX classes touch hardware or model files.

## Models

Downloaded by `scripts/download_models.py` into `JARVIS_MODELS_DIR` and
pinned by SHA-256 in `src/jarvis/audio/models.py`:

- `silero_vad.onnx` (Silero VAD v5)
- `melspectrogram.onnx`, `embedding_model.onnx`, `hey_jarvis_v0.1.onnx` (openWakeWord v0.5.1)

A different wake phrase is a different classifier file; the two shared
stages stay the same.

## Threads

See `ARCHITECTURE.md` section 5. In short: the capture callback only pushes
into the ring buffer, the worker thread runs the pipeline, the playback
callback drains a queue without ever blocking, and Python polls. No native
thread takes the GIL.

## Tests

`native/tests/audio/` covers every class. Tests that need model files skip
themselves when the files are absent; nothing in the suite opens a real
device. The hardware path is verified by `scripts/record_test_audio.py` and
the `hardware`-marked Python tests; the full engine on recorded speech
(real VAD, segmenter and barge-in, no microphone) is exercised by
`scripts/benchmark_latency.py` through `ReplaySource`.
