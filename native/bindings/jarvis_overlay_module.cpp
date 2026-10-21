/// @file jarvis_overlay_module.cpp
/// @brief Python module @c jarvis_overlay.
///
/// A thin layer over jarvis::overlay: the window runs its own UI thread and
/// never calls back into Python, so every setter is a plain, cheap forward.
/// Only construction and close() release the GIL, because they wait for
/// that thread.

#include "jarvis/common/errors.hpp"
#include "jarvis/overlay/errors.hpp"
#include "jarvis/overlay/layout.hpp"
#include "jarvis/overlay/model.hpp"
#include "jarvis/overlay/window.hpp"

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "module_info.hpp"

#include <string>
#include <string_view>

namespace py = pybind11;
using namespace jarvis::overlay;  // NOLINT(google-build-using-namespace) - binding file only

namespace {

// Translators are consulted most-recently-registered first, so the base
// class goes first and the derived class after it.
void defineExceptions(const py::module_& module) {
    const auto& nativeError =
        py::register_exception<jarvis::NativeError>(module, "NativeError", PyExc_RuntimeError);
    py::register_exception<OverlayError>(module, "OverlayError", nativeError.ptr());
}

void defineEnums(const py::module_& module) {
    py::enum_<AssistantState>(module, "AssistantState",
                              "What Jarvis is doing; mirrors the Python state machine.")
        .value("IDLE", AssistantState::Idle)
        .value("LISTENING", AssistantState::Listening)
        .value("THINKING", AssistantState::Thinking)
        .value("SPEAKING", AssistantState::Speaking);

    py::enum_<Corner>(module, "Corner", "Screen corner the panel is anchored to.")
        .value("TOP_LEFT", Corner::TopLeft)
        .value("TOP_RIGHT", Corner::TopRight)
        .value("BOTTOM_LEFT", Corner::BottomLeft)
        .value("BOTTOM_RIGHT", Corner::BottomRight);
}

void defineConfig(const py::module_& module) {
    py::class_<ModelLimits>(module, "ModelLimits",
                            "Bounds that keep the overlay's memory independent of its input.")
        .def(py::init<>())
        .def_readwrite("level_history", &ModelLimits::levelHistory)
        .def_readwrite("max_text_chars", &ModelLimits::maxTextChars)
        .def_readwrite("wake_flash", &ModelLimits::wakeFlashDuration,
                       "datetime.timedelta the wake-word flash stays visible.");

    py::class_<OverlayConfig>(module, "OverlayConfig",
                              "Geometry and behaviour of the window (sizes at 96 DPI).")
        .def(py::init<>())
        .def_readwrite("corner", &OverlayConfig::corner)
        .def_readwrite("width", &OverlayConfig::width)
        .def_readwrite("height", &OverlayConfig::height)
        .def_readwrite("margin", &OverlayConfig::margin)
        .def_readwrite("click_through", &OverlayConfig::clickThrough)
        .def_readwrite("opacity", &OverlayConfig::opacity)
        .def_readwrite("active_fps", &OverlayConfig::activeFps)
        .def_readwrite("idle_fps", &OverlayConfig::idleFps)
        .def_readwrite("title", &OverlayConfig::title)
        .def_readwrite("limits", &OverlayConfig::limits);
}

void defineSnapshot(const py::module_& module) {
    py::class_<OverlaySnapshot>(module, "OverlaySnapshot",
                                "Copy of what the UI thread draws; for tests and the dashboard.")
        .def_readonly("state", &OverlaySnapshot::state)
        .def_readonly("levels", &OverlaySnapshot::levels)
        .def_readonly("transcript", &OverlaySnapshot::transcript)
        .def_readonly("response", &OverlaySnapshot::response)
        .def_readonly("visible", &OverlaySnapshot::visible)
        .def_readonly("wake_flash", &OverlaySnapshot::wakeFlash)
        .def_readonly("revision", &OverlaySnapshot::revision);
}

void setStateByName(OverlayWindow& window, std::string_view name) {
    const auto state = parseAssistantState(name);
    if (!state.has_value()) {
        throw py::value_error("unknown assistant state '" + std::string{name} +
                              "'; expected IDLE, LISTENING, THINKING or SPEAKING");
    }
    window.setState(*state);
}

void defineWindow(const py::module_& module) {
    py::class_<OverlayWindow>(module, "OverlayWindow",
                              "Always-on-top status panel rendered on a native UI thread.")
        .def(py::init<OverlayConfig>(), py::arg("config") = OverlayConfig{},
             py::call_guard<py::gil_scoped_release>(),
             "Create the window and wait until it is on screen. Raises OverlayError.")
        .def("set_state", py::overload_cast<AssistantState>(&OverlayWindow::setState),
             py::arg("state"))
        .def("set_state", &setStateByName, py::arg("name"),
             "Same as above, by member name (\"LISTENING\"); ValueError if unknown.")
        .def("push_level", &OverlayWindow::pushLevel, py::arg("level"),
             "Append one level sample in [0, 1] to the waveform.")
        .def("set_transcript", &OverlayWindow::setTranscript, py::arg("text"))
        .def("set_response", &OverlayWindow::setResponse, py::arg("text"))
        .def("set_visible", &OverlayWindow::setVisible, py::arg("visible"))
        .def("flash_wake_word", &OverlayWindow::flashWakeWord)
        .def("visible", &OverlayWindow::visible)
        .def("state", &OverlayWindow::state)
        .def("running", &OverlayWindow::running)
        .def("snapshot", &OverlayWindow::snapshot)
        .def("config", &OverlayWindow::config, py::return_value_policy::copy)
        .def("close", &OverlayWindow::close, py::call_guard<py::gil_scoped_release>(),
             "Close the window and join the UI thread. Idempotent.")
        .def("__enter__", [](OverlayWindow& self) -> OverlayWindow& { return self; })
        .def(
            "__exit__",
            [](OverlayWindow& self, const py::object& /*type*/, const py::object& /*value*/,
               const py::object& /*traceback*/) {
                const py::gil_scoped_release release;
                self.close();
            },
            py::arg("type"), py::arg("value"), py::arg("traceback"));
}

}  // namespace

PYBIND11_MODULE(jarvis_overlay, module) {
    module.doc() =
        "Status overlay: always-on-top Dear ImGui window with waveform, state "
        "indicator and transcript. See native/overlay/README.md.";
    jarvis::bindings::defineModuleInfo(module, "implemented in step 13");

    defineExceptions(module);
    defineEnums(module);
    defineConfig(module);
    defineSnapshot(module);
    defineWindow(module);
}
