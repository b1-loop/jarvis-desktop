# Contributing

Jarvis is a solo portfolio project, but it is run like a team codebase: every
change goes through a branch, CI and a pull request. This document is the
onboarding guide for anyone (including future me) touching the code.

## Prerequisites

| Tool | Version | Why |
| --- | --- | --- |
| Python | 3.12.x | pinned in `pyproject.toml`; 3.13 is not yet supported by all audio deps |
| uv | ≥ 0.5 | dependency management and virtualenv |
| CMake | ≥ 3.28 | native build |
| Ninja | any | fast native builds, required by the CMake presets |
| MSVC | VS 2022 17.10+ or VS 2026 | C++20 compiler on Windows |
| LLVM | 23 | `clang-tidy`, and `clang-format` for editors (the hook pins its own 23.1.3) |
| sccache | any | only to reproduce CI locally with the `ci-windows` preset (`winget install Mozilla.sccache`) |

## First-time setup

```powershell
git clone --recurse-submodules https://github.com/b1-loop/jarvis-desktop.git
cd jarvis-desktop
uv sync --all-extras --group dev                 # Python deps + dev tools
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
uv run python scripts/build_native.py            # builds native/ into src/jarvis/_native/, runs Catch2
uv run python scripts/download_models.py --local # VAD + wake word; --local adds whisper.cpp, llama.cpp, Piper
ollama pull qwen2.5:7b                           # the default LLM (install Ollama first)
copy .env.example .env                           # optional keys: Claude, integrations
```

`scripts/build_native.py` options:

| Flag | Effect |
| --- | --- |
| `--preset windows-release` (default), `windows-debug`, `ci-windows` | CMake preset; `ci-windows` adds sccache and a shared FetchContent cache |
| `--skip-tests` | build only, do not run the Catch2 suite |
| `--clean` | delete the build tree first |
| `--tidy` | build, test, then run clang-tidy on every file |
| `--tidy-only` | configure and lint without building |
| `--tidy-changed-since <ref>` | lint only the `.cpp` files changed since `<ref>` (all files when a header or CMake file changed) |
| `--tidy-shard i/n`, `--jobs N` | lint one interleaved shard, with N parallel processes (what CI does) |

`scripts/download_models.py` verifies every file by SHA-256 and skips
files that already match; `--force` re-downloads, `--models-dir` changes
the target (default `JARVIS_MODELS_DIR`).

Run the assistant:

```powershell
uv run jarvis --text-only          # terminal conversation, no audio needed
uv run jarvis --mode cloud         # full voice pipeline
uv run jarvis --mode local         # fully offline
```

If the checkout already exists, fetch the native submodules with
`git submodule update --init --recursive`; the native build fails with a
clear message when `native/third_party/` is empty.

## Workflow

1. Branch from `main`: `feat/<slug>`, `fix/<slug>`, `docs/<slug>`, `chore/<slug>`.
2. Make the change **with tests**. Coverage must stay above 80 %.
3. Run the full check locally (pre-commit does most of it automatically):

   ```powershell
   uv run ruff check . ; uv run ruff format --check . ; uv run mypy ; uv run pytest -m "not hardware"
   uv run python scripts/build_native.py --tidy   # native build, Catch2 tests, clang-tidy
   ```

   Test markers (`pyproject.toml`): `native` needs the compiled modules and
   skips without them; `hardware` opens a real microphone or speakers and
   never runs in CI (`uv run pytest -m hardware` by hand); `integration`
   wires several modules with mocked I/O. CI's Linux lane runs
   `-m "not native and not hardware"`, the Windows lane the `native` tests
   after the C++ build. Coverage must stay above 80 %.

4. Commit with [Conventional Commits](https://www.conventionalcommits.org/):
   `feat(audio): add ring buffer`, `fix(agent): handle empty tool result`, `docs: add ADR-0003`.
5. Push and open a pull request with the output of the checks above.

**CI status.** `.github/workflows/ci.yml` runs on every push and pull
request (the sections below describe its caches and clang-tidy shards). Run
the pre-commit hooks and the commands in step 3 locally first, plus the
native build when C++ changed; a pull request is merged once CI passes.

## Tool versions: one source of truth

The pre-commit hooks, a terminal and CI must agree on formatter and linter
versions, otherwise two versions fight over the same lines (ruff 0.7 and 0.17
disagreed about which `noqa` comments were needed, and code was bent around
that). So each tool has exactly one pin:

| Tool | Pinned in | Used by |
| --- | --- | --- |
| ruff, mypy | `uv.lock` | `uv run ...` in a terminal, the `repo: local` hooks in `.pre-commit-config.yaml`, CI |
| clang-format | `.pre-commit-config.yaml` (`mirrors-clang-format` 23.1.3) | the hook and CI's `clang-format check`, which runs that hook; LLVM 23 on the dev machine matches the major version |
| clang-tidy | the LLVM installation | local `--tidy`, and the runner image's LLVM in CI (printed by the `clang-tidy` job) |

To upgrade ruff or mypy, bump them with `uv lock --upgrade-package ruff`; the
hooks follow automatically. To upgrade clang-format, change the hook `rev`,
run `uv run pre-commit run clang-format --all-files`, and commit the result
on its own.

## CI caches

The Windows native lane spent most of its time on work that does not change
between pushes. Four caches cut it down; none of them skips a check.

| Cache | What | Key | Why it is safe |
| --- | --- | --- | --- |
| FetchContent | miniaudio, the ONNX Runtime zip, Catch2 and Dear ImGui sources in `native/build/_fetchcontent` (`FETCHCONTENT_BASE_DIR` in the `ci-windows` preset) | hash of the three `CMakeLists.txt` that declare FetchContent, falling back to the newest cache | FetchContent re-checks every dependency against its declaration (URL hash, git tag) and downloads again when it changed |
| sccache | compiled objects, including whisper.cpp, llama.cpp and ggml | content hash of each compile, stored in the GitHub Actions cache | the compiler output for identical input is identical; a miss just compiles. `CMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded` (`/Z7`) keeps objects cacheable should a target request debug info, because `/Zi` writes a shared PDB that sccache cannot cache |
| ONNX models | the audio models (`scripts/download_models.py` without `--local`) | hash of `src/jarvis/audio/models.py` | the script verifies each file's SHA-256 and re-downloads a mismatch |
| uv | wheels for `uv sync`, on every job | `uv.lock` (setup-uv default) | `UV_FROZEN=1` installs exactly the lock file. setup-uv v4 talked to the retired cache API and never restored anything ("Cache service responded with 400"), hence the bump |

LOCAL-mode models (whisper.cpp and llama.cpp, 570 MB) are not downloaded in CI;
the model-backed inference tests skip themselves when the files are absent,
while the wrappers' logic is covered by the model-free tests.

clang-tidy is the long pole: about 40 s per translation unit, 42 minutes for
the whole tree in one serial process. It now runs in four `clang-tidy (i/4)`
jobs parallel to the build, each linting an interleaved quarter of the files
with one process per CPU (`--tidy-shard`, `--jobs`). On a pull request only
the `.cpp` files the PR changed are linted (`--tidy-changed-since HEAD^1`),
unless the PR touches a header, a CMake file, a submodule, the presets,
`.clang-tidy` or `build_native.py`; then every file is. Pushes to `main`
always lint every file, so a PR's narrowed run is backed by a full one when it
lands. Locally, `--tidy` keeps building first and lints everything;
`--tidy-only` configures and lints without building, and
`--tidy-changed-since origin/main` mirrors the PR behaviour.

To reproduce the CI build locally, install sccache and run
`uv run python scripts/build_native.py --preset ci-windows`. Day-to-day work
uses `windows-release` or `windows-debug`, which need neither.

## Running the benchmark

`scripts/benchmark_latency.py` speaks fixed English and Swedish prompts
into the real voice pipeline and prints the README's latency table:

```powershell
$env:JARVIS_STT_DEVICE = "cpu"   # only if the CUDA libraries are missing
uv run python scripts/benchmark_latency.py                   # cloud-like and local, 3 runs per prompt, silent
uv run python scripts/benchmark_latency.py --modes cloud --languages en --runs 10
uv run python scripts/benchmark_latency.py --sink device     # play the answers through the speakers
```

The default null sink "plays" answers at the real-time rate without sound,
so the benchmark never talks over the person at the computer. Every stage
up to the hand-off to the device is identical; real playback adds one
device period (about 10 ms) before the sound is audible.

Prompts are synthesised once with edge-tts and cached under
`data/bench/prompts/`; a `jarvis_audio.ReplayFeed` replaces the microphone,
so the native VAD, segmenter and barge-in run unchanged. It needs Ollama for
the cloud-like mode, `download_models.py --local` for the local mode, the
native build, and the network for edge-tts. Results go to
`data/bench/results.json` (ignored by version control); a full run takes about 20 minutes.
`scripts/benchmark_local.py` is the narrower provider-by-provider
comparison behind ADR-0013.

## Code rules

The full list lives in `CLAUDE.md`. The ones people trip on most:

- Functions under 30 lines, one responsibility each.
- Google-style docstrings that say **why**, on every public symbol.
- No concrete implementation imported across layers; depend on the interface in `base.py`.
- `requires_confirmation = True` on any tool that changes the world.
- Stubs are named `Stub*` and documented as stubs. Never silently fake a feature.
- No secrets anywhere but `.env`. `gitleaks` runs in pre-commit and CI.

## Adding a tool

Create `src/jarvis/tools/<name>.py` with a class deriving from `jarvis.agent.tool.Tool`,
define its pydantic input model, implement `async run()`, and add
`tests/tools/test_<name>.py`. The registry discovers it automatically; there is
nothing to register.

## Adding a native component

Add the sources under `native/<component>/`, a Catch2 test under `native/tests/`,
a pybind11 binding under `native/bindings/`, and a thin Python wrapper with a
pure-Python fallback or a clear `NativeUnavailableError`. Public C++ APIs carry
Doxygen comments. Run `scripts/build_native.py --tidy` (or
`--tidy-only --tidy-changed-since origin/main`) before the PR.

## Working on the desktop app

`jarvis --background` (what the shortcuts run) has no console: output goes
to `data/logs/jarvis.log`, and a second launch only pokes the first one.
While developing, keep using `uv run jarvis` or `--text-only` in a terminal,
and quit any autostarted instance from its tray menu first: the single-
instance lock only guards windowless launches, so a terminal session next
to it would open the microphone a second time. Point a test run at a
scratch directory with `JARVIS_DATA_DIR` to get its own lock and log. Code
in `src/jarvis/app/` must import on Linux (CI): keep Windows-only calls
behind `sys.platform` checks and their tests behind `skipif`. After
changing the state palette, regenerate the icon with
`uv run python scripts/make_icon.py` (a test fails if it is stale).

## Architecture decisions

Anything that a reviewer might ask "why?" about gets an ADR in `docs/adr/`.
Copy `docs/adr/template.md`, number it, and link it from `docs/adr/README.md`.
