# Status overlay: always-on-top Dear ImGui window on DirectX 11.
#
# Dear ImGui is fetched at configure time from the official repository and
# compiled as its own static library with the library's own warning level;
# only our code is held to /W4 /WX. The Win32 and DX11 backends are part of
# that library so the overlay target itself stays free of backend sources.

include(FetchContent)

if(NOT WIN32)
    message(FATAL_ERROR "The overlay renders through DirectX 11 and the Win32 window API; "
                        "a Metal/Vulkan backend is needed before it can build elsewhere.")
endif()

FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.9b
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(imgui)

# --- Dear ImGui core + backends, compiled once as third-party code ---------
add_library(jarvis_imgui STATIC
    "${imgui_SOURCE_DIR}/imgui.cpp"
    "${imgui_SOURCE_DIR}/imgui_draw.cpp"
    "${imgui_SOURCE_DIR}/imgui_tables.cpp"
    "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_win32.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_dx11.cpp")
target_include_directories(jarvis_imgui SYSTEM PUBLIC
    "${imgui_SOURCE_DIR}"
    "${imgui_SOURCE_DIR}/backends")
# Obsolete APIs are disabled so a future ImGui bump fails loudly at compile
# time instead of silently changing behaviour.
target_compile_definitions(jarvis_imgui PUBLIC
    IMGUI_DISABLE_OBSOLETE_FUNCTIONS
    IMGUI_DISABLE_DEMO_WINDOWS)
# The DX11 backend compiles its shaders at runtime through d3dcompiler.
target_link_libraries(jarvis_imgui PUBLIC d3d11 dxgi d3dcompiler)

# --- The overlay -----------------------------------------------------------
add_library(jarvis_overlay_ui STATIC
    src/layout.cpp
    src/model.cpp
    src/panel.cpp
    src/window.cpp)
add_library(jarvis::overlay ALIAS jarvis_overlay_ui)

target_include_directories(jarvis_overlay_ui PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
# Public headers never include ImGui or Win32; both stay behind the pimpl in
# window.cpp and the private panel.hpp, hence the PRIVATE link.
target_link_libraries(jarvis_overlay_ui
    PUBLIC jarvis::common
    PRIVATE jarvis_imgui)
# Keep <windows.h> from defining min/max macros and from dragging in the
# legacy APIs; this only affects our translation units, not ImGui's.
target_compile_definitions(jarvis_overlay_ui PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
jarvis_set_warnings(jarvis_overlay_ui)
