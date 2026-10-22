"""The three views of the assistant: overlay, tray icon and dashboard.

- ``overlay``: facade over the native Dear ImGui panel (``jarvis_overlay``),
  always on top, with the state, a live waveform and the current exchange.
  Native because it redraws at frame rate (ADR-0011).
- ``tray``: notification-area icon coloured by state, with a menu whose
  actions are handed back to the asyncio loop (pystray, ADR-0015).
- ``dashboard``: tkinter window with the transcript, tool calls, latency
  statistics and a memory inspector, fed by ``view_model`` (ADR-0015).

All three are pure consumers of the event bus; they never drive the
pipeline, and each degrades to a ``Stub*`` stand-in when there is no
desktop or no native build.
"""
