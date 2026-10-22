# 0015. Tray icon with pystray, dashboard window with tkinter

Date: 2026-10-22
Status: Accepted

## Context

Step 13 asks for two more views of the assistant next to the native status
overlay (ADR-0011):

- a **tray icon** that always shows the current state and offers a few
  actions (show the dashboard, toggle the overlay, mute the microphone,
  quit);
- a **dashboard window** for a developer or a curious user: live state,
  rolling transcript, tool-call log with durations and confirmation
  outcomes, per-stage latency statistics, and a memory inspector.

Both must be *pure consumers of the event bus*: they may not drive the
pipeline, slow it down, or crash it. Both must degrade to nothing on a
headless machine (the Linux CI runner, text-only mode) and the modules must
import there. The overlay is C++ because a 60 fps waveform redrawn in
Python would compete with STT and the LLM stream for the GIL. The tray
redraws once per state change and the dashboard a few tables a few times a
second, so that argument does not automatically carry over.

## Decision

**Shared view model.** `jarvis.ui.view_model.DashboardModel` subscribes to
the bus on the asyncio loop and folds events into display state. It emits
frozen `DashboardSnapshot` values with per-section revision counters. Every
presentation decision lives there as plain data: the state palette (the
overlay's `Theme`, in hex), truncation, time and duration formats, the
tool-call row lifecycle (awaiting confirmation, declined, running, ok,
error), the tray tooltip and the memory sections. The memory tab reads
through a `MemoryInspector` Protocol that `SqliteStore` satisfies
structurally, so the UI never imports the concrete store.

**Tray: pystray + Pillow.** `TrayIcon` pre-renders one 64 px Pillow image per
state and runs a pystray icon on a daemon thread of its own. On
`StateChanged` it swaps the image and tooltip. Menu clicks arrive on the
tray thread and are forwarded with `loop.call_soon_threadsafe` to an
injected `TrayCallbacks` Protocol; the tray holds no business logic. The
mute check mark is set by the application through `set_microphone_muted`
after it actually muted, so it cannot disagree with the audio engine.
pystray is behind a `TrayBackend` Protocol and is imported lazily inside
the factory, because it chooses a platform backend at import time and
raises on a headless Linux box.

**Dashboard: standard-library tkinter.** `Dashboard` creates the Tk
interpreter, every widget and their destruction on one dedicated thread
running its own `mainloop`. The asyncio side never calls Tk: it puts
snapshots and commands (show, hide, stop) into a `queue.Queue`, which the
Tk thread drains from an `after()` timer every 100 ms, rendering only the
newest snapshot and only the sections whose revision moved. The one
reverse path, the memory "Refresh" button, hops onto the loop with
`call_soon_threadsafe`. Closing the window hides it; the tray reopens it.

![Dashboard driven by scripts/demo_ui.py](../media/dashboard.png)

**Graceful degradation.** `create_tray` and `create_dashboard` are async
factories that start the real surface and await its readiness through a
future (never blocking the loop). A missing pystray, an unsupported
desktop, a backend that fails to start, a Tcl/Tk `TclError` ("no
display") or a start timeout logs a warning and returns `StubTray` /
`StubDashboard`, which keep tracking state so text mode and tests still
see it. Neither factory is wired into the composition root yet; that
belongs to the voice pipeline step.

## Alternatives considered

- **Dear ImGui through the native overlay module** – one UI technology for
  everything and no GIL cost. But tables, scrolling text, tabs and a
  refresh button in immediate-mode C++ would add a few hundred lines of
  C++, a second window to the overlay's thread, and a native rebuild for
  every column change, for a window that is opened occasionally and
  updates a few times a second. It would also disappear whenever the
  native build is missing, which is exactly when a diagnostic window is
  most useful.
- **Local web dashboard (HTTP + WebSocket + browser)** – rich layout and
  charts, and remote viewing. It needs a server dependency, a port, an
  origin/auth story so another local process cannot read the transcript
  and memory, and a browser tab that is not "a window of the app".
  Reasonable later if the dashboard grows charts; not worth the attack
  surface now.
- **Qt (PySide6)** – the most capable toolkit, but a 100+ MB dependency,
  its own event loop that wants the main thread (or qasync), and LGPL
  packaging questions, for four tables.
- **Dear PyGui / wxPython** – more modern look than Tk, but each is a new
  native dependency with its own threading rules; tkinter ships with the
  interpreter we already use (python.org and uv-managed builds alike).
- **Tk on the main thread with the asyncio loop on a worker** – the usual
  Tk advice, but it inverts the application: the loop is the program and
  the dashboard is optional. A dedicated Tk thread keeps the dashboard
  removable without touching how the assistant starts.
- **Polling the model from Tk instead of pushing snapshots** – would make
  the Tk thread read state the loop is mutating. Immutable snapshots
  through a queue keep the ownership one-directional, the same shape as the
  overlay's setters plus per-frame snapshot.
- **pystray's `run_detached` or `Icon.run` on the main thread** – the main
  thread belongs to asyncio; `run` on our own daemon thread works on
  Windows (pystray creates its window in `run`) and keeps the lifecycle
  explicit.

## Consequences

- The dashboard costs the event loop one snapshot build per event and
  nothing else; Tk's interpreter loop sleeps between polls and releases
  the GIL while idle.
- Tk is thread-affine, so `_DashboardView` must never escape its thread.
  The thread runs `gc.collect()` after destroying the window so widget
  reference cycles are not finalised on another thread, which aborts Tcl.
- On Windows, Tcl intermittently fails to read its init scripts while
  pytest redirects file descriptors 0-2; the Tk tests suspend output
  capture. Headless CI skips them via a fixture that tries `tkinter.Tk()`.
- Tool calls and confirmations are correlated by tool name, newest first,
  which is correct while the agent runs one turn's tools sequentially. If
  tools ever run concurrently, the events need a call id.
- Provider and integration status is not shown because no event carries it
  yet; adding a status event in `core/events.py` makes it one handler and
  one field in the view model.
- The tray, overlay and dashboard share one palette, so the three views of
  the same state look the same.
