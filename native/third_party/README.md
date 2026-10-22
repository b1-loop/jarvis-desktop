# third_party

Git submodules built from source by `native/inference` for LOCAL mode:

| Submodule | Pinned at | Used for |
| --- | --- | --- |
| `whisper.cpp` | `v1.9.5` | offline speech-to-text (`WhisperTranscriber`) |
| `llama.cpp` | commit `0a1ef309` | offline text generation (`LlamaGenerator`); also provides the shared ggml |

The pins are the commits recorded in the superproject; update one with
`git -C native/third_party/<name> checkout <tag>` and commit the new
submodule pointer together with any wrapper changes. Fetch them in an
existing checkout with `git submodule update --init --recursive`; the
native build stops with a clear message when this directory is empty.

Nothing here is linted or formatted by our tools: `.clang-format` and
`.clang-tidy` exclude it, and the submodules compile with their own warning
levels as `SYSTEM` includes (see `native/inference/README.md`).
