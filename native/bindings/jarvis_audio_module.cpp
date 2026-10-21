/// @file jarvis_audio_module.cpp
/// @brief Python module @c jarvis_audio.
///
/// Scaffold: exposes version and status only. Audio engine arrives in step 3.

#include <pybind11/pybind11.h>

#include "module_info.hpp"

PYBIND11_MODULE(jarvis_audio, module) {
    module.doc() =
        "Audio engine: capture, ring buffer, VAD, wake word, playback with barge-in.\n\nScaffold: "
        "the audio engine arrives in step 3.";
    jarvis::bindings::defineModuleInfo(module, "scaffold: audio engine arrives in step 3");
}
