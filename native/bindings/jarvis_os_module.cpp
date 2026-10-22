/// @file jarvis_os_module.cpp
/// @brief Python module @c jarvis_os.
///
/// A thin layer: converts strings and byte buffers, releases the GIL around
/// every call that touches the desktop, COM or WMI, and maps the native
/// exception hierarchy onto Python classes. All behaviour lives in jarvis::os.

#include "jarvis/common/errors.hpp"
#include "jarvis/os/audio_endpoint.hpp"
#include "jarvis/os/clipboard.hpp"
#include "jarvis/os/errors.hpp"
#include "jarvis/os/hotkey.hpp"
#include "jarvis/os/notifier.hpp"
#include "jarvis/os/process_control.hpp"
#include "jarvis/os/screenshot.hpp"
#include "jarvis/os/system_controls.hpp"
#include "jarvis/os/text_injector.hpp"
#include "jarvis/os/window_manager.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "module_info.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace jarvis::os;  // NOLINT(google-build-using-namespace) - binding file only

namespace {

using GilRelease = py::call_guard<py::gil_scoped_release>;

py::bytes toBytes(const std::vector<std::uint8_t>& data) {
    return {std::string(data.begin(), data.end())};
}

// Translators are consulted most-recently-registered first, so the base
// class goes first and the derived classes after it.
void defineExceptions(const py::module_& module) {
    const auto& nativeError = py::register_local_exception<jarvis::NativeError>(
        module, "NativeError", PyExc_RuntimeError);
    const auto& osError =
        py::register_local_exception<OsError>(module, "OsError", nativeError.ptr());
    py::register_local_exception<UnsupportedError>(module, "UnsupportedError", osError.ptr());
    py::register_local_exception<WindowNotFoundError>(module, "WindowNotFoundError", osError.ptr());
    py::register_local_exception<ProcessNotFoundError>(module, "ProcessNotFoundError",
                                                       osError.ptr());
}

void defineHotkeys(py::module_& module) {
    py::class_<Hotkey>(module, "Hotkey", "A parsed key combination.")
        .def(py::init<>())
        .def_readwrite("modifiers", &Hotkey::modifiers, "Bitmask of Win32 MOD_* flags.")
        .def_readwrite("virtual_key", &Hotkey::virtualKey, "Win32 virtual-key code.")
        .def(
            "__eq__", [](const Hotkey& a, const Hotkey& b) { return a == b; }, py::is_operator())
        .def("__repr__", [](const Hotkey& hotkey) {
            return "Hotkey(modifiers=" + std::to_string(hotkey.modifiers) +
                   ", virtual_key=" + std::to_string(hotkey.virtualKey) + ")";
        });

    module.def("parse_hotkey", &parseHotkey, py::arg("spec"),
               "Parse 'ctrl+alt+j' style text. Raises ValueError on a malformed spec.");
    module.def("format_hotkey", &formatHotkey, py::arg("hotkey"),
               "Canonical lower-case spelling of a hotkey.");

    py::class_<HotkeyEvent>(module, "HotkeyEvent", "One press of a registered hotkey.")
        .def_readonly("id", &HotkeyEvent::id)
        .def_readonly("hotkey", &HotkeyEvent::hotkey);

    py::class_<HotkeyManager>(module, "HotkeyManager",
                              "System-wide hotkeys on a dedicated message-loop thread.")
        .def(py::init<>(), GilRelease())
        .def("register_hotkey", &HotkeyManager::registerHotkey, py::arg("hotkey"), GilRelease(),
             "Register and return the id used in HotkeyEvent. Raises OsError if taken.")
        .def("unregister_hotkey", &HotkeyManager::unregisterHotkey, py::arg("id"), GilRelease())
        .def("poll_events", &HotkeyManager::pollEvents,
             "Drain the presses reported since the previous poll.")
        .def("registered_count", &HotkeyManager::registeredCount);
}

void defineWindows(const py::module_& module) {
    py::class_<WindowInfo>(module, "WindowInfo", "A top-level window.")
        .def_readonly("handle", &WindowInfo::handle)
        .def_readonly("title", &WindowInfo::title)
        .def_readonly("process_name", &WindowInfo::processName)
        .def_readonly("executable_path", &WindowInfo::executablePath)
        .def_readonly("pid", &WindowInfo::pid)
        .def("__repr__", [](const WindowInfo& info) {
            return "WindowInfo(handle=" + std::to_string(info.handle) + ", title='" + info.title +
                   "', process_name='" + info.processName + "', pid=" + std::to_string(info.pid) +
                   ")";
        });

    py::class_<Win32WindowManager>(module, "WindowManager", "Active window and window management.")
        .def(py::init<>())
        .def("active_window", &Win32WindowManager::activeWindow, GilRelease())
        .def("list_windows", &Win32WindowManager::listWindows, GilRelease())
        .def("describe", &Win32WindowManager::describe, py::arg("handle"), GilRelease())
        .def("focus", &Win32WindowManager::focus, py::arg("handle"), GilRelease())
        .def("focus_by_title", &Win32WindowManager::focusByTitle, py::arg("substring"),
             GilRelease())
        .def("minimize", &Win32WindowManager::minimize, py::arg("handle"), GilRelease())
        .def("maximize", &Win32WindowManager::maximize, py::arg("handle"), GilRelease())
        .def("restore", &Win32WindowManager::restore, py::arg("handle"), GilRelease())
        .def("close", &Win32WindowManager::close, py::arg("handle"), GilRelease());
}

void defineProcesses(const py::module_& module) {
    py::class_<Win32ProcessControl>(module, "ProcessControl", "Launch and close applications.")
        .def(py::init<>())
        .def("launch", &Win32ProcessControl::launch, py::arg("target"), py::arg("arguments") = "",
             GilRelease(), "Open an executable, document or URL; returns the pid or 0.")
        .def("close", &Win32ProcessControl::close, py::arg("pid"), py::arg("force") = false,
             GilRelease())
        .def("close_by_name", &Win32ProcessControl::closeByName, py::arg("executable_name"),
             py::arg("force") = false, GilRelease());
}

void defineInputAndNotifications(const py::module_& module) {
    py::class_<SendInputTextInjector>(module, "TextInjector", "Type text at the cursor.")
        .def(py::init<>())
        .def("type_text", &SendInputTextInjector::typeText, py::arg("text"),
             py::arg("press_enter") = false, GilRelease());

    py::class_<BalloonNotifier>(module, "Notifier", "Toast notifications via the tray icon.")
        .def(py::init<std::string_view>(), py::arg("tooltip") = "Jarvis", GilRelease())
        .def("notify", &BalloonNotifier::notify, py::arg("title"), py::arg("message"),
             GilRelease());

    py::class_<Win32Clipboard>(module, "Clipboard", "Plain-text clipboard access.")
        .def(py::init<>())
        .def("text", &Win32Clipboard::text, GilRelease())
        .def("set_text", &Win32Clipboard::setText, py::arg("text"), GilRelease());
}

void defineSystem(const py::module_& module) {
    py::class_<Win32SystemControls>(module, "SystemControls",
                                    "Master volume, display brightness and the lock screen.")
        .def(py::init<>())
        .def("volume", &Win32SystemControls::volume, GilRelease())
        .def("set_volume", &Win32SystemControls::setVolume, py::arg("percent"), GilRelease())
        .def("muted", &Win32SystemControls::muted, GilRelease())
        .def("set_muted", &Win32SystemControls::setMuted, py::arg("muted"), GilRelease())
        .def("brightness", &Win32SystemControls::brightness, GilRelease(),
             "Raises UnsupportedError when no display offers software brightness.")
        .def("set_brightness", &Win32SystemControls::setBrightness, py::arg("percent"),
             GilRelease())
        .def("lock_screen", &Win32SystemControls::lockScreen, GilRelease());

    py::class_<GdiScreenshotter>(module, "Screenshotter", "Screen capture as PNG bytes.")
        .def(py::init<>())
        .def(
            "capture_screen",
            [](const GdiScreenshotter& self, bool allMonitors) {
                std::vector<std::uint8_t> png;
                {
                    const py::gil_scoped_release release;
                    png = self.captureScreen(allMonitors);
                }
                return toBytes(png);
            },
            py::arg("all_monitors") = false)
        .def(
            "capture_window",
            [](const GdiScreenshotter& self, WindowHandle handle) {
                std::vector<std::uint8_t> png;
                {
                    const py::gil_scoped_release release;
                    png = self.captureWindow(handle);
                }
                return toBytes(png);
            },
            py::arg("handle"));
}

void defineAudioEndpoint(py::module_& module) {
    py::class_<AudioEndpointInfo>(module, "AudioEndpoint",
                                  "The default output device: friendly name and form factor.")
        .def_readonly("name", &AudioEndpointInfo::name)
        .def_property_readonly(
            "form_factor",
            [](const AudioEndpointInfo& self) {
                return std::string{formFactorName(self.formFactor)};
            },
            "Core Audio form factor in snake_case: 'speakers', 'headphones', 'headset', ...");
    module.def("default_output_endpoint", &defaultOutputEndpoint, GilRelease(),
               "The default output device, or None when there is none or it cannot be read.");
}

}  // namespace

PYBIND11_MODULE(jarvis_os, module) {
    module.doc() =
        "OS integration for Windows: global hotkeys, windows, processes, text injection, "
        "notifications, volume, brightness, lock screen, screenshot and clipboard. "
        "See native/os_integration/README.md.";
    jarvis::bindings::defineModuleInfo(module, "implemented in step 4");

    defineExceptions(module);
    defineHotkeys(module);
    defineWindows(module);
    defineProcesses(module);
    defineInputAndNotifications(module);
    defineSystem(module);
    defineAudioEndpoint(module);
}
