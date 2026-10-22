# 0025. A launch while running replays the HUD; a wake brings a closed HUD back

Date: 2026-10-22
Status: Accepted

## Context

Jarvis is started from the desktop shortcut (`jarvis --background`,
ADR-0021). When it already runs, the second launch hands over to the
running instance through the single-instance socket and exits. Until now
that hand-over only opened the dashboard. A user who had closed the HUD
(the hover "close" control) or minimized it to the orb got no HUD at all,
and the start-up boot animation could never be seen again: the native
model (`OverlayModel::playBoot`) started it on the first request of the
process only.

Closing had a second problem. The close control hid the HUD until the
tray's "Toggle overlay" was used. A user who closed it and then said "Hey
Jarvis" heard the wake beep but saw nothing, which reads as broken.

The user asked for two things: opening Jarvis should show "the whole
animation", and a dedicated "Jarvis HUD" shortcut should do exactly that.

## Decision

**A launch while running brings the whole HUD forward and replays the full
boot.** The hand-over carries a request, `HandoverRequest.SHOW` (the plain
shortcut) or `HandoverRequest.SHOW_HUD` (`--show-hud`, the new "Jarvis
HUD" shortcut). Both call `HudActions.bring_forward`, which shows a closed
HUD, collapses the expanded panel (the boot hands off into the corner
HUD), restores the orb (and remembers that, like the tray's "Restore
HUD"), re-reads the MODEL / MODE / MIC readout off the event loop and
calls `Overlay.replay_boot`. `SHOW` also opens the dashboard, as before, so
nothing the shortcut used to do is lost; `SHOW_HUD` does not. The boot
spans the work area, so re-placing the window for it re-asserts
`HWND_TOPMOST` and the HUD ends up in front without stealing focus.

**The native boot can be replayed, deliberately.** `OverlayModel` keeps
`playBoot` first-request-only, because the launcher and the voice front
end both ask for the start-up boot, and gains a separate `replayBoot`:
it moves the boot's start to now, clears a skip and replaces the readout
(bounded like the first). Every boot effect is a pure function of the time
since that start, so nothing else needs resetting. When Jarvis is not Idle
the replay is skipped at once, as leaving Idle already skips a running
boot: the HUD glides into its corner instead of covering a conversation.
Skipping is unchanged: `skip_boot()` or any state other than Idle jumps to
the hand-off. (There is no click-to-skip; the boot is click-through by
design, ADR-0022.)

**A first launch always plays the full boot.** Closing the HUD is never
persisted, so every new process starts with a visible HUD and boots. A
persisted "minimized" choice is kept: the native window already plays the
boot in the full HUD and shrinks into the orb after the hand-off; a test
now covers that for the replay too.

**Closing means "hide until needed".** The overlay bridge shows a closed
HUD again on the next `WakeWordHeard`, which both the wake word and the
voice hotkey publish, with the ordinary wake ripple. The boot stays for
launches: the user is about to talk and the HUD must be ready to listen.
An orb the user chose stays an orb and is only made visible. Hiding
because a full-screen application is in front is native and does not
change visibility, so a wake does not fight it. The close control's
tooltip says "Hide until needed (waking Jarvis brings it back)".

**A second shortcut, "Jarvis HUD".** `scripts/install_shortcuts.py`
writes it on the desktop and in the Start menu (`--no-hud` removes it),
with its own icon: the same generated mark in the listening colour
(`jarvis-hud.ico`, regenerated with `jarvis.ico` by `scripts/make_icon.py`).
It runs the same `uvw` command plus `--show-hud`, which implies
`--background`; with no Jarvis running it simply starts one. The shortcut
manifest is now keyed by (kind, location); the Jarvis shortcut keeps the
bare location key, so manifests written before this change still load.

## Alternatives considered

- **Replay through `playBoot` by lifting its "once" rule.** Simpler API,
  but the two start-up callers would then play the boot twice; a separate,
  explicit call keeps that bug impossible.
- **Let the model ignore a replay while a boot plays.** It would need the
  boot's duration, which only the animation timeline knows. A restart is
  harmless (double-clicking the shortcut restarts the intro) and keeps the
  model free of animation timing.
- **Replay the boot over the orb and shrink back.** The native window
  supports it, but launching Jarvis is asking to see it, so the HUD is
  restored. The wake path keeps the orb because talking does not need the
  full HUD.
- **Full boot on a wake as well.** Four seconds of intro before Jarvis
  shows it is listening would slow every turn after a close; the ripple
  is the established "I heard you" signal.
- **Drop the dashboard from the plain relaunch.** That would change what
  the existing shortcut has always done; users who want the HUD alone now
  have the "Jarvis HUD" shortcut.
- **Persist "closed" across restarts.** A HUD that never appears after a
  restart would look like a broken install, and the user asked for the
  boot on every launch.

## Consequences

- Every launch while running is visible: the HUD and its intro, and the
  dashboard for the plain shortcut.
- Hiding the HUD from the tray is also undone by the next wake; that menu
  entry is now "hide until needed" in practice, which matches the close
  control.
- The hand-over protocol has a second command. An instance from before
  this change rejects `show-hud`, so the new shortcut falls back to the
  "already running" notification until that instance is restarted.
- The user's shortcuts must be reinstalled once to get "Jarvis HUD":
  `uv run python scripts/install_shortcuts.py` (add `--autostart` to keep
  an autostart shortcut).
