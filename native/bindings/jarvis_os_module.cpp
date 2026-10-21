/// @file jarvis_os_module.cpp
/// @brief Python module @c jarvis_os.
///
/// Scaffold: exposes version and status only. OS integration layer arrives in step 4.

#include <pybind11/pybind11.h>

#include "module_info.hpp"

PYBIND11_MODULE(jarvis_os, module) {
    module.doc() =
        "OS integration: hotkeys, windows, text injection, notifications, system "
        "controls.\n\nScaffold: the OS integration layer arrives in step 4.";
    jarvis::bindings::defineModuleInfo(module, "scaffold: OS integration layer arrives in step 4");
}
