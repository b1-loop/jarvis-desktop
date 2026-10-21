# overlay

Always-on-top Dear ImGui window on DirectX 11 showing waveform, state indicator and live transcript. Driven purely by events from the Python side.

Lands in step 13. Until then `JARVIS_BUILD_OVERLAY` defaults to OFF in
`native/CMakeLists.txt` and the matching Python module exposes only
`__version__` and `IMPLEMENTATION_STATUS`.
