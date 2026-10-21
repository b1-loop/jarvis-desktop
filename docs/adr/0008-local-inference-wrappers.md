# 0008. Own thin C++ wrappers over whisper.cpp and llama.cpp

Date: 2026-10-21
Status: Accepted

## Context

LOCAL mode needs offline speech-to-text and an offline LLM. whisper.cpp and
llama.cpp are the de-facto CPU inference engines for both, and each already
has a maintained Python package (`whisper-cpp-python`, `llama-cpp-python`)
that could be added to `pyproject.toml` in one line.

Those packages, however, build (or download) their own copies of ggml,
expose the raw C API through ctypes, and run their generation loops in
Python. Jarvis has four native components in one CMake tree, a documented
rule that native threads never touch the interpreter, and a story to tell:
the hot paths are C++ and Python conducts.

The streaming shape also matters. Speech synthesis should start on the
first sentence, barge-in must be able to abort a generation mid-token, and
the audio engine already proved that a native worker plus a polling Python
side keeps latency predictable.

## Decision

whisper.cpp and llama.cpp are **git submodules** under `native/third_party`,
pinned to release tags, built from source as static libraries in the same
CMake tree as the other components and wrapped in our own thin API:

- `WhisperTranscriber` and `LlamaGenerator` use only the stable C headers
  (`whisper.h`, `llama.h`) behind a pimpl, so third-party types never leak
  into other headers and a submodule bump touches two `.cpp` files.
- One ggml: llama.cpp's copy is built, whisper.cpp links against it. Both
  projects explicitly support being embedded this way.
- `LlamaGenerator` streams: each sampled token is converted to text and
  handed to a callback before the next decode. Sampling goes through the
  `llama_sampler` chain API (top-k, top-p, temperature) and prompts are
  rendered with the model's own chat template via
  `llama_chat_apply_template`.
- Stop strings and partial UTF-8 are handled in a pure `StopMatcher`, so a
  stop sequence split across tokens is caught and Python never receives a
  broken character.
- The Python binding is **pull-based**: `generate()` starts the loop on a
  native thread and returns a `GenerationStream`; `next_token()` blocks
  with the GIL released, `cancel()` raises a `CancellationToken`. The
  native thread never acquires the GIL, exactly like the audio engine.
- **CPU only** for now: `GGML_NATIVE=OFF` gives a portable AVX2 build that
  runs on the CI runner and on any recent x86-64 laptop. CUDA is a build
  option away but would make the native lane depend on a GPU toolkit.

## Alternatives considered

- **`llama-cpp-python` / `whisper-cpp-python`** – quickest to integrate,
  but a second ggml and a second build system (their wheels or their
  scikit-build) next to ours, generation loops under the GIL, and nothing
  to show for the "C++ muscle" half of the project.
- **Push-based streaming (native thread calls a Python callback)** – the
  obvious API, but every token would re-acquire the GIL from a native
  thread, interleaving with the audio engine's polling and the asyncio
  loop. Pull-based keeps the thread rules uniform across modules.
- **Blocking generate() that returns the full text** – simplest binding,
  but TTS could not start until the answer is complete; the point of the
  wrapper is latency.
- **Vendoring source snapshots instead of submodules** – no `--recurse-
  submodules` in the clone, but upgrades become opaque multi-thousand-line
  diffs. Submodules pinned to tags keep provenance visible.
- **GPU build from day one** – faster, but CUDA on a GitHub runner is not
  available and the CPU build already serves the 0.5B–3B models LOCAL mode
  targets.

## Consequences

- A clone needs `--recurse-submodules` (or `git submodule update --init`);
  the native build now takes several minutes more on a cold tree.
- `JARVIS_BUILD_INFERENCE` is ON by default; switching it OFF builds a
  stub module and skips the submodules entirely.
- The submodule versions are visible in `jarvis_inference.WHISPER_VERSION`
  and `LLAMA_VERSION`.
- The Python `STT` and `LLMClient` implementations for LOCAL mode consume
  this API and are tracked under step 12; ADR-0009 will document the
  CLOUD vs LOCAL trade-offs once both are in place.
