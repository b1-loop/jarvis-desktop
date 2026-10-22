"""Jarvis as a desktop application: windowless start, one instance, shortcuts.

- ``windowless``: detects a launch without a console (``uvw``, ``pythonw``
  or ``--background``) and turns terminal output into log events.
- ``background``: the windowless session. File logging, single instance,
  Ollama on demand, voice mode with the tray as the only visible surface,
  and an error tray instead of a silent exit when voice cannot start.
- ``single_instance``: an OS file lock plus a loopback hand-over so a
  second launch brings the running assistant forward instead of starting
  another one.
- ``ollama_launcher``: starts the local Ollama server when it is not
  answering, hidden and detached.
- ``shortcuts``: desktop, Start-menu and Startup-folder shortcuts that
  launch the windowless app (``scripts/install_shortcuts.py``).
- ``icon``: the application icon, drawn from the tray's own artwork.

ADR-0021 records why Jarvis is launched through ``uvw`` and shortcuts
rather than shipped as a frozen executable.
"""
