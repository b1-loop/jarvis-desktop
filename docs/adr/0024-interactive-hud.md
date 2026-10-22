# 0024. Interactive HUD: hit-tested shapes, explicit focus, polled UI events, user settings

Date: 2026-10-22
Status: Accepted

## Context

Until now the overlay was display-only (ADR-0011, 0022, 0023): a
click-through window that Python drove through cheap setters. The user
wants to use it: close it, type to Jarvis, change settings and open it
larger, and move it freely (any part of the HUD's body drags, edges snap,
it works across monitors and DPI changes and is remembered).

Constraints that still hold: the HUD sits over whatever the user is doing,
so transparent pixels must stay click-through and showing it must never
steal the keyboard; the native UI thread must never call into Python (GIL,
frame pacing); business logic stays in Python; the ADR-0023 frame budget and
the 15 fps idle rate stay; both surfaces (DirectComposition and the layered
LWA fallback) must work.

## Decision

**Hit-testing: a per-frame hit map plus WS_EX_TRANSPARENT toggled by a
cursor poll.** Each frame the window builds a `HitMap` (hud_layout.hpp) from
the same layout it draws: the ring's disc and the text column's scrim drag,
the hover controls, the text field and the quit confirmation are controls;
in the expanded panel the frame drags, the tabs, body and input row are
controls, the corner grip resizes. `WM_NCHITTEST` answers `HTTRANSPARENT`
outside the map and `HTCLIENT` inside. That alone is not enough:
`HTTRANSPARENT` only passes a click to windows of the *same thread*, so
over another application the click would be eaten. The window therefore
stays `WS_EX_TRANSPARENT` (truly click-through) and the UI thread polls
`GetCursorPos` once per frame; only while the cursor is over a shape is
`WS_EX_TRANSPARENT` cleared, and a `WM_NCHITTEST` outside the map sets it
again immediately. The composition window keeps `WS_EX_LAYERED` (alpha 255)
throughout so only one style bit flips.

**Dragging by SetCapture, not HTCAPTION.** Answering `HTCAPTION` would
hand the drag to `DefWindowProc`'s modal move loop, which stops this
thread's frame loop for the whole drag (a frozen HUD) and makes snapping,
Alt and DPI handling awkward. The window captures the mouse instead and
places the panel from raw cursor positions: the grab point is kept as a
share of the size (so it stays under the cursor when a DPI change resizes
the panel mid-drag), edges within 16 logical px snap to the work area of the
monitor under most of the panel (Alt disables it), and the result is
clamped fully onto that monitor. The position is saved as monitor device
name + logical offset from its work area (`placement.hpp`), restored at the
next start on that monitor if it is still connected (else the primary, then
clamped), and re-resolved on `WM_DISPLAYCHANGE`. A double-click on the ring
resets to the default corner. The expanded panel has its own position and
size.

**Focus model: never on show, explicitly for typing.** The window keeps
`WS_EX_NOACTIVATE` and answers `MA_NOACTIVATE`, so clicks on controls never
activate it. Clicking the text field (or the keyboard control) remembers the
foreground window, then calls `SetForegroundWindow`/`SetFocus` (allowed:
this process received the last input). Enter or Escape gives focus back to
the remembered window; clicking elsewhere simply takes it.

**Native-to-Python: a polled event queue.** User actions become typed
`UiEvent`s (`SubmitText`, `CloseRequested`, `QuitRequested`, `ToggleMute`,
`SetExpanded`, `SettingChanged`, `OpenLog`, `RestartRequested`,
`SettingsOpened`, `PlacementChanged`) in a bounded `UiEventQueue`:
adjacent `SettingChanged` for the same key and `PlacementChanged` for the
same panel coalesce, and a full queue drops its oldest event and counts it.
Python's overlay bridge drains it every 30 ms on the event loop and
dispatches each event to one small handler (`core/hud_wiring.py`). In the
other direction Python pushes data: the settings page (rows with kind,
choices and current value), history lines, mute state and appearance. The
native side renders and edits them and reports edits as events; it decides
nothing.

**Typing shares the voice conversation.** `CombinedInput` is the
orchestrator's input in voice mode: typed while idle starts a turn
(cancelling the pending wake-word wait); typed while thinking or speaking
cancels the turn and silences speech, then the typed request is handled
(typed barge-in; the speaker is stopped only after the turn has unwound to
IDLE, so the voice barge-in controller does not mistake it for speech);
typed during a confirmation answers it. "Speak replies" switches each
answer between the TTS output and a silent text output.

**User settings: a pydantic-settings source over
`data_dir/user_settings.json`, precedence init kwargs > environment
variables > user_settings.json > .env > defaults.** Environment variables
win because they are explicit for one launch (tests, CI, a debugging
session) and must not be silently overridden by a file the HUD wrote weeks
ago. The file wins over `.env` because `.env` is the installer's baseline
and the file holds the user's later, deliberate choices; otherwise changing
the voice in the HUD would appear not to work after a restart. Each value
is validated against its field alone, so a bad entry is logged and skipped
instead of stopping the start; secrets are never read from it. Changes
apply live where the session can (appearance, voice with edge-tts, speak
replies, wake-word threshold through a new native setter, start with
Windows); the rest (LLM provider, Ollama model, speech language, wake word
on/off) is marked "applies after restart", and the windowless app's Restart
relaunches itself after releasing the single-instance lock.

## Alternatives considered

- **HTTRANSPARENT only** - simple, but clicks over other processes' windows
  are swallowed (Windows passes them only within the thread).
- **Per-pixel hit-testing with UpdateLayeredWindow** - automatic
  click-through on alpha 0, but it means a CPU copy of every frame and
  drops the DirectComposition surface and its GPU bloom (ADR-0022/0023).
- **HTCAPTION dragging** - free system move, but the modal loop freezes
  the render loop and Aero Snap's rules are not the HUD's.
- **Callbacks from the UI thread into Python** - lower latency, but the UI
  thread would need the GIL and could stall a frame behind the event loop;
  30 ms polling is below a noticeable delay and costs one lock when idle.
- **A Tk or web settings window** - another toolkit and another look; the
  settings page in the HUD keeps the holographic style and one renderer.
- **Writing the user's choices into .env** - .env is hand-edited and may
  hold comments and secrets; a separate, machine-written file is safer.

## Consequences

- Measured with `scripts/demo_overlay.py --interactive` on the development
  machine (AMD Radeon 890M, 2560 x 1600 at 150 %, large panel,
  DirectComposition with bloom), medians of three runs while the machine
  was in normal use (single phases varied by up to 4 ms):

  | Phase | CPU / frame | GPU / frame | Rate |
  | --- | --- | --- | --- |
  | idle, hover controls off | 1.07 ms | 0.39 ms | 15 fps |
  | idle, hover controls shown | 2.59 ms | 0.39 ms | 15 fps (demo hook) |
  | typing and the scripted answer | 1.35 ms | 0.42 ms | 60 fps |
  | expanded view (history) | 3.98 ms | 0.72 ms | 15 fps |

  With the cursor elsewhere the interactive layer adds one `GetCursorPos`
  and a hit test of about a dozen shapes per frame, and the idle HUD stays
  at 15 fps. A real hover (cursor over the HUD) switches to the active rate
  for responsiveness; the demo hook used for the table keeps the idle rate.
  The expanded view costs most (ImGui lays out the whole history each frame)
  and only exists while the user looks at it.
- The window takes the keyboard only on an explicit click; automated tests
  and the demo use hooks (`inject_ui_event`, `set_demo`) and never move the
  mouse or take focus.
- New risk: an unknown event kind from a mismatched build raises
  `HudProtocolError` in the bridge instead of being ignored.
- Not done: the tray menu item is still "Toggle overlay" (it shows a hidden
  HUD again); the history keeps 500 lines per session; the IME candidate
  window follows ImGui's default positioning.
