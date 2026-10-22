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
    IMGUI_USE_WCHAR32
    IMGUI_DISABLE_DEMO_WINDOWS)
# The DX11 backend compiles its shaders at runtime through d3dcompiler.
target_link_libraries(jarvis_imgui PUBLIC d3d11 dxgi d3dcompiler)

# --- Shaders ---------------------------------------------------------------
# The HLSL passes are embedded as text into a generated header at configure
# time and compiled with D3DCompile when the window is created (the design
# notes in src/shaders.cpp say why not fxc at build time). The shared
# prelude is prepended here so D3DCompile needs no include handler.
set(JARVIS_OVERLAY_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(JARVIS_OVERLAY_SHADERS
    fullscreen_vs:kFullscreenVs
    bloom_prefilter_ps:kBloomPrefilterPs
    bloom_down_ps:kBloomDownPs
    bloom_up_ps:kBloomUpPs
    composite_ps:kCompositePs)
file(READ "${JARVIS_OVERLAY_SHADER_DIR}/common.hlsli" jarvis_shader_common)
set(JARVIS_OVERLAY_SHADER_DEFINITIONS "")
set(jarvis_shader_inputs "${JARVIS_OVERLAY_SHADER_DIR}/common.hlsli")
foreach(entry IN LISTS JARVIS_OVERLAY_SHADERS)
    string(REPLACE ":" ";" parts "${entry}")
    list(GET parts 0 shader_file)
    list(GET parts 1 shader_symbol)
    set(shader_path "${JARVIS_OVERLAY_SHADER_DIR}/${shader_file}.hlsl")
    file(READ "${shader_path}" shader_text)
    list(APPEND jarvis_shader_inputs "${shader_path}")
    string(APPEND JARVIS_OVERLAY_SHADER_DEFINITIONS
        "inline constexpr std::string_view ${shader_symbol} = R\"hlsl(${jarvis_shader_common}\n${shader_text})hlsl\";\n\n")
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${jarvis_shader_inputs})
set(JARVIS_OVERLAY_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
configure_file("${JARVIS_OVERLAY_SHADER_DIR}/embedded_shaders.hpp.in"
    "${JARVIS_OVERLAY_GENERATED_DIR}/jarvis/overlay/embedded_shaders.hpp" @ONLY)

# --- The overlay -----------------------------------------------------------
add_library(jarvis_overlay_ui STATIC
    src/animation.cpp
    src/animator.cpp
    src/effects.cpp
    src/hud.cpp
    src/hud_boot.cpp
    src/hud_data.cpp
    src/hud_draw.cpp
    src/hud_expanded.cpp
    src/hud_expanded_ui.cpp
    src/hud_icons.cpp
    src/hud_interact.cpp
    src/hud_layout.cpp
    src/hud_panel.cpp
    src/hud_scene.cpp
    src/hud_settings_ui.cpp
    src/layout.cpp
    src/model.cpp
    src/particles.cpp
    src/placement.cpp
    src/postfx.cpp
    src/scene.cpp
    src/shaders.cpp
    src/surface.cpp
    src/system_info.cpp
    src/text_entry.cpp
    src/ui_events.cpp
    src/window.cpp
    src/window_input.cpp
    src/window_place.cpp)
add_library(jarvis::overlay ALIAS jarvis_overlay_ui)

target_include_directories(jarvis_overlay_ui
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${JARVIS_OVERLAY_GENERATED_DIR}")
# Public headers never include ImGui or Win32; both stay behind the pimpl in
# window.cpp and the private hud/surface headers, hence the PRIVATE link.
# dcomp provides DirectComposition for the per-pixel transparent surface,
# d3dcompiler the post-processing shaders, ole32 the Core Audio query for
# the boot readout's microphone name.
target_link_libraries(jarvis_overlay_ui
    PUBLIC jarvis::common
    PRIVATE jarvis_imgui dcomp d3dcompiler ole32 propsys shcore)
# Keep <windows.h> from defining min/max macros and from dragging in the
# legacy APIs; this only affects our translation units, not ImGui's.
target_compile_definitions(jarvis_overlay_ui PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
jarvis_set_warnings(jarvis_overlay_ui)
