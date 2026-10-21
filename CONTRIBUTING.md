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
| LLVM | ≥ 18 | `clang-format` and `clang-tidy` |

## First-time setup

```powershell
git clone --recurse-submodules https://github.com/b1-loop/jarvis-desktop.git
cd jarvis
uv sync --all-extras                     # Python deps + dev tools
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
uv run python scripts/build_native.py    # builds native/ into src/jarvis/_native/
uv run python scripts/download_models.py # VAD, wake word and (optionally) local models
copy .env.example .env                   # then fill in your keys
```

Run the assistant:

```powershell
uv run jarvis --text-only          # terminal conversation, no audio needed
uv run jarvis --mode cloud         # full voice pipeline
uv run jarvis --mode local         # fully offline
```

## Workflow

1. Branch from `main`: `feat/<slug>`, `fix/<slug>`, `docs/<slug>`, `chore/<slug>`.
2. Make the change **with tests**. Coverage must stay above 80 %.
3. Run the full check locally (pre-commit does most of it automatically):

   ```powershell
   uv run ruff check . ; uv run ruff format --check . ; uv run mypy ; uv run pytest
   uv run python scripts/build_native.py --tidy   # native build, Catch2 tests, clang-tidy
   ```

4. Commit with [Conventional Commits](https://www.conventionalcommits.org/):
   `feat(audio): add ring buffer`, `fix(agent): handle empty tool result`, `docs: add ADR-0003`.
5. Push and open a pull request. CI must be green before merging.

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
Doxygen comments. `clang-tidy` runs on every PR.

## Architecture decisions

Anything that a reviewer might ask "why?" about gets an ADR in `docs/adr/`.
Copy `docs/adr/template.md`, number it, and link it from `docs/adr/README.md`.
