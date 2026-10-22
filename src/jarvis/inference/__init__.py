"""Model manifest and native loader for LOCAL-mode inference.

The C++ side (``native/inference``) wraps whisper.cpp and llama.cpp; this
package pins the model files those wrappers (and the Piper voice) are
tested against and loads ``jarvis_inference`` with a remedy-bearing error.
The ``STT``, ``LLMClient`` and ``TTS`` implementations that use them live in
``jarvis.stt.whisper_cpp``, ``jarvis.llm.local_llama_client`` and
``jarvis.tts.piper``.
"""
