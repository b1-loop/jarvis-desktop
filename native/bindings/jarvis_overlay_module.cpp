/// @file jarvis_overlay_module.cpp
/// @brief Python module @c jarvis_overlay.
///
/// Scaffold: exposes version and status only. Overlay arrives in step 13.

#include <pybind11/pybind11.h>

#include "module_info.hpp"

PYBIND11_MODULE(jarvis_overlay, module) {
    module.doc() =
        "Status overlay: always-on-top Dear ImGui window with waveform and "
        "transcript.\n\nScaffold: the overlay arrives in step 13.";
    jarvis::bindings::defineModuleInfo(module, "scaffold: overlay arrives in step 13");
}
