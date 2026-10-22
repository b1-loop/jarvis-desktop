/// @file jarvis_inference_scaffold.cpp
/// @brief Python module @c jarvis_inference.
///
/// Built only when JARVIS_BUILD_INFERENCE is OFF: exposes version and status so the
/// Python side can tell a deliberately disabled build from a broken one.

#include <pybind11/pybind11.h>

#include "module_info.hpp"

PYBIND11_MODULE(jarvis_inference, module) {
    module.doc() =
        "Local inference: whisper.cpp speech-to-text and llama.cpp text generation.\n\nScaffold: "
        "built with JARVIS_BUILD_INFERENCE=OFF.";
    jarvis::bindings::defineModuleInfo(module, "disabled: built with JARVIS_BUILD_INFERENCE=OFF");
}
