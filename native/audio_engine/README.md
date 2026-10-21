# audio_engine

Microphone capture, lock-free ring buffer, Silero VAD and openWakeWord via ONNX Runtime, playback with a cancellation token for barge-in. Audio I/O through miniaudio.

Lands in step 3. Until then `JARVIS_BUILD_AUDIO` defaults to OFF in
`native/CMakeLists.txt` and the matching Python module exposes only
`__version__` and `IMPLEMENTATION_STATUS`.
