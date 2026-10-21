/// @file jarvis_inference_module.cpp
/// @brief Python module @c jarvis_inference.
///
/// Scaffold: exposes version and status only. Inference wrappers arrive in step 12.

#include <pybind11/pybind11.h>

#include "module_info.hpp"

PYBIND11_MODULE(jarvis_inference, module) {
    module.doc() =
        "Local inference: whisper.cpp speech-to-text and llama.cpp text generation.\n\nScaffold: "
        "the inference wrappers arrive in step 12.";
    jarvis::bindings::defineModuleInfo(module, "scaffold: inference wrappers arrive in step 12");
}
