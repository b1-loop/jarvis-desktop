# os_integration

Win32 implementation of the platform services tools need: global hotkeys, active window and window management, text injection via SendInput, toast notifications, volume (Core Audio), brightness (WMI), lock screen, screenshot.

Lands in step 4. Until then `JARVIS_BUILD_OS` defaults to OFF in
`native/CMakeLists.txt` and the matching Python module exposes only
`__version__` and `IMPLEMENTATION_STATUS`.
