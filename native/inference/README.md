# inference

Thin RAII wrappers over whisper.cpp and llama.cpp (git submodules under third_party/) exposing streaming transcription and token generation with the GIL released.

Lands in step 12. Until then `JARVIS_BUILD_INFERENCE` defaults to OFF in
`native/CMakeLists.txt` and the matching Python module exposes only
`__version__` and `IMPLEMENTATION_STATUS`.
