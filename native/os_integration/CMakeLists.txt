# OS integration: Win32 hotkeys, windows, processes, text injection,
# notifications, volume, brightness, lock screen, screenshot and clipboard.
#
# Everything here is Windows-only by design (CLAUDE.md: Windows 11 first,
# other platforms behind the Python Platform interface). The public headers
# under include/ contain no Win32 types; <windows.h> is confined to src/.

if(NOT WIN32)
    message(FATAL_ERROR "os_integration is the Win32 backend; a macOS/Linux backend would be a "
                        "sibling directory selected here by platform.")
endif()

add_library(jarvis_os_integration STATIC
    src/clipboard.cpp
    src/hotkey_manager.cpp
    src/hotkey_parser.cpp
    src/message_loop_thread.cpp
    src/notifier.cpp
    src/process_control.cpp
    src/screenshot.cpp
    src/system_controls.cpp
    src/text_encoding.cpp
    src/text_injector.cpp
    src/win32_util.cpp
    src/window_manager.cpp)
add_library(jarvis::os_integration ALIAS jarvis_os_integration)

target_include_directories(jarvis_os_integration PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(jarvis_os_integration PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")

# Lean <windows.h>, no min/max macros, wide-character API by default, and a
# Windows 10 1607 baseline for SetThreadDpiAwarenessContext.
target_compile_definitions(jarvis_os_integration PRIVATE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    UNICODE
    _UNICODE
    _WIN32_WINNT=0x0A00)

target_link_libraries(jarvis_os_integration
    PUBLIC jarvis::common
    PRIVATE
        user32      # windows, hotkeys, SendInput, clipboard
        gdi32       # screenshot
        ole32       # COM, IStream
        oleaut32    # BSTR / VARIANT for WMI
        shell32     # ShellExecuteEx, Shell_NotifyIcon
        dwmapi      # cloaked windows, extended frame bounds
        windowscodecs  # WIC PNG encoder
        wbemuuid    # WMI class ids
        kernel32)
jarvis_set_warnings(jarvis_os_integration)
