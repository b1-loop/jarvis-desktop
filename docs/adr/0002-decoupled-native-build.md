# 0002. Build native extensions separately from the Python package

Date: 2026-10-21
Status: Accepted

## Context

The project mixes a Python package with four pybind11 extension modules
(`jarvis_audio`, `jarvis_os`, `jarvis_inference`, `jarvis_overlay`). The usual
answer is `scikit-build-core` as the build backend so that `pip install` or
`uv sync` compiles everything. That couples every Python workflow, including
text-only development, linting and the Linux CI jobs, to a working C++
toolchain, CMake, ONNX Runtime and, in LOCAL mode, whisper.cpp and llama.cpp
submodules that take minutes to compile.

Most of the Python code never needs the native modules: the agent loop, tools,
memory and LLM clients are pure Python and run in `--text-only` mode.

## Decision

The Python package uses `hatchling` and is pure Python. Native modules are
built by `scripts/build_native.py` (a thin wrapper over the CMake presets)
and copied into `src/jarvis/_native/`, which is git-ignored.

`jarvis._native` exposes an `available_modules()` helper and loads each
extension lazily. Python wrappers in `jarvis/audio`, `jarvis/os_integration`
and so on import through that shim and raise a clear `NativeUnavailableError`
(or fall back where a pure-Python fallback makes sense) when the `.pyd` is
missing.

## Alternatives considered

- **scikit-build-core** – idiomatic for single-extension libraries, but makes
  `uv sync` on a laptop without MSVC fail outright, and forces the Linux CI
  jobs to compile Win32 code or skip the package entirely.
- **Separate wheels per native module** – clean, but overkill for a project
  that is run from a checkout, not published to PyPI.
- **ctypes against plain DLLs** – avoids pybind11, loses type safety and the
  numpy buffer protocol that the audio path depends on.

## Consequences

- Text-only development and the fast CI lane need no compiler.
- The Windows CI lane proves the native build and the import path together.
- Two build commands instead of one; `CONTRIBUTING.md` documents both.
- If the project is ever published as a wheel, this decision should be
  revisited in favour of `scikit-build-core` with the native build gated by an
  environment variable.
