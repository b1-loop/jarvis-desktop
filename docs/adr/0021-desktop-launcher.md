# 0021. Desktop launcher: uvw and shortcuts, not a frozen executable (yet)

Date: 2026-10-22
Status: Accepted

## Context

Until now Jarvis started from a terminal (`uv run jarvis`). The first
question from the user who runs it every day was "do I have to go to
PowerShell every time?". A voice assistant should behave like any other
Windows desktop app: a Start-menu entry, optionally starting with Windows,
no console window, one instance, and a tray icon as the only surface until
the user asks for more.

Constraints:

- The runtime is a uv-managed checkout with native extensions built
  separately (ADR-0002) and models downloaded under `models/`.
- Without a console (`pythonw`), `sys.stdout`/`sys.stderr` are `None`: a
  single `print`, `input()` or log line to stderr kills the process before
  anyone sees why.
- A process the user cannot see must still explain failures (no native
  build, no models, no microphone, Ollama down) instead of vanishing.

## Decision

**Launch the checkout through `uvw`, from Windows shortcuts.**
`scripts/install_shortcuts.py` creates `.lnk` files on the desktop, in the
Start menu and (with `--autostart`) in the user's Startup folder. Each runs
`uvw.exe run --project <checkout> jarvis --background` in the checkout
(falling back to the checkout's `.venv\Scripts\pythonw.exe -m jarvis
--background` when uv is not on PATH). The shortcuts are written with the
`WScript.Shell` COM object through PowerShell; the files created are
recorded in `data/desktop-shortcuts.json`, and `--uninstall` deletes
exactly those.

**`--background` (or no stdout) selects a windowless session**
(`jarvis.app.background`):

- logs to a size-capped rotating file, `data/logs/jarvis.log` (JSON lines);
- takes a single-instance lock (`data/jarvis.lock`, an OS file lock); a
  second launch asks the first, over a loopback socket with a random token,
  to show its dashboard, and exits (since ADR-0025 it also brings the HUD
  forward and replays its boot, and a "Jarvis HUD" shortcut asks for the
  HUD alone);
- starts Ollama hidden if it is the LLM and is not answering;
- runs voice mode with the dashboard hidden and the tray icon as the only
  visible surface, and shows one notification on start;
- on failure, notifies the reason and the log path, keeps a tray icon whose
  tooltip carries the error, and exits on Quit.

The icon (`src/jarvis/app/assets/jarvis.ico`, 16-256 px) is generated from
the tray's own drawing by `scripts/make_icon.py`.

## Alternatives considered

- **Frozen executable (PyInstaller, Nuitka).** The "real app" answer and
  the roadmap item, but not yet: it needs a build pipeline that bundles the
  pybind11 `.pyd` files, ONNX Runtime, CTranslate2 and Tk, and every build
  is hundreds of MB that must be rebuilt for each change. An unsigned
  executable also triggers SmartScreen's "Windows protected your PC"
  warning, and many antivirus products flag PyInstaller bootloaders. A code
  signing certificate and an installer (MSIX or Inno Setup) come with it.
- **A Run key in the registry for autostart.** Invisible to most users and
  easy to leave behind. A Startup-folder shortcut is visible in Explorer
  and in Task Manager's Startup apps, and removing it is deleting a file.
- **A Windows service.** Services run in session 0 without access to the
  user's desktop, tray or (by default) microphone; wrong model for an
  interactive assistant.
- **pywin32 or comtypes for the `.lnk` files.** A native dependency for
  three shortcuts created once; PowerShell and `WScript.Shell` ship with
  Windows.
- **A named mutex (`CreateMutexW`) for single instance.** Equivalent on
  Windows, but untestable on the Linux CI runner; the file lock behaves the
  same on both, and the OS releases it when the process dies, so a crash
  never leaves a stale lock.

## Consequences

- No build step: a `git pull` and `uv sync` update the app, and shortcuts
  keep working because they point at the checkout. No unsigned binary, so
  no SmartScreen warning.
- The app needs uv (or the synced `.venv`) and the checkout on disk; moving
  the checkout breaks the shortcuts until the script is run again. `uv run`
  syncs before starting, which adds a moment to each start.
- Everything a console would have shown (greeting, transcript, warnings)
  now goes to the log file; the tray tooltip and notifications are the
  user-facing channel for problems.
- Native balloon notifications can chime (the notifier respects quiet
  hours but sets no "no sound" flag); there is one notification per start.
- Roadmap: a frozen, signed build with an installer replaces the uvw
  shortcuts once the native build and model download can be packaged.
