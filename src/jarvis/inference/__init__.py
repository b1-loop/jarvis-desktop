"""Model manifest for the native inference module (``jarvis_inference``).

The C++ side (``native/inference``) wraps whisper.cpp and llama.cpp; this
package pins the model files those wrappers are tested against. The
``STT`` and ``LLMClient`` implementations that drive the wrappers in LOCAL
mode live in ``jarvis.stt`` and ``jarvis.llm`` and arrive with the Python
half of step 12.
"""
