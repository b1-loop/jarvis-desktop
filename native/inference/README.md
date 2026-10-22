# inference

Thin RAII wrappers over [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
(offline speech-to-text) and [llama.cpp](https://github.com/ggml-org/llama.cpp)
(offline text generation). Exposed to Python as `jarvis_inference`. Why this
exists instead of the ready-made Python bindings: `docs/adr/0010-local-inference-wrappers.md`.

## Components

| Header | Class / function | Role |
| --- | --- | --- |
| `whisper_transcriber.hpp` | `WhisperTranscriber` | load a ggml model, `transcribe(pcm16k, options) -> Transcript{text, language, segments}` |
| `llama_generator.hpp` | `LlamaGenerator` | load a GGUF model, `generate(prompt or messages, options, callback, cancel)` streaming one piece per token |
| `generation_stream.hpp` | `GenerationStream` | runs a generation on a native thread; the consumer pulls pieces with `next()` |
| `chat_template.hpp` | `ChatMessage`, `renderChatTemplate` | role/content messages rendered through `llama_chat_apply_template` |
| `stop_matcher.hpp` | `StopMatcher`, `completeUtf8Prefix` | stop strings across token boundaries, partial UTF-8 held back |
| `backend.hpp` | `ensureBackend` | one-time ggml init, INFO logs silenced |
| `common/cancellation_token.hpp` | `CancellationToken` | atomic "please stop" flag shared with Python |

Every public header hides the third-party types behind a pimpl, so nothing
that includes `jarvis/inference/*.hpp` sees `whisper.h` or `llama.h`.

## Build

`native/inference/CMakeLists.txt` adds the two submodules under
`native/third_party/` as static libraries:

- llama.cpp is added first and brings its own ggml; whisper.cpp is added
  second and, because a `ggml` target already exists, skips its own copy.
  This is the mechanism both projects provide for embedding ("otherwise
  assume ggml is added by a parent CMakeLists.txt") and avoids two CPU
  backends in one process.
- CPU only (`GGML_NATIVE=OFF`, portable AVX2 build), no CUDA. The
  `gpuLayers` option exists so the API does not change when a GPU backend
  is enabled.
- Examples, tools, tests, servers and networking of both projects are off;
  the submodules compile with their own warning levels and are consumed as
  `SYSTEM` includes, so our `/W4 /WX` policy applies only to our files.
- The submodules are compiled as C++17 (their declared standard); our code
  stays C++20.

Whole-tree builds take a few minutes; `JARVIS_BUILD_INFERENCE=OFF` builds a
stub `jarvis_inference` module that only reports its status.

## Models

Pinned by revision and SHA-256 in `src/jarvis/inference/models.py` and
fetched by `scripts/download_models.py --local`:

- `ggml-tiny.bin` (whisper tiny, multilingual, 75 MB)
- `qwen2.5-0.5b-instruct-q4_k_m.gguf` (Qwen2.5 0.5B Instruct, Apache-2.0, 490 MB)

These are the smallest models that exercise the full code path; LOCAL mode
proper can point at any ggml / GGUF file.

## Threads and the GIL

`WhisperTranscriber::transcribe` blocks on the calling thread; the Python
binding releases the GIL around it. `LlamaGenerator::generate` delivers
tokens through a callback on the calling thread. The Python binding never
calls back into Python from native code: `LlamaGenerator.generate(...)`
returns a `GenerationStream` whose producer thread fills a queue, and
`next_token()` blocks with the GIL released until a piece arrives, `None`
at the end. `cancel()` raises a `CancellationToken` the producer checks
before every token. This mirrors the polling design of `jarvis_audio`
(ARCHITECTURE.md section 5).

## Tests

`native/tests/inference/` covers the pure logic (stop matcher, UTF-8
boundaries, chat templates, the stream, the token) without models, and the
two wrappers with the small models above; those tests skip when the files
are absent. `tests/inference/test_native_inference.py` checks the Python
surface the same way.
