# Header-only helpers shared by every native component.

add_library(jarvis_common INTERFACE)
add_library(jarvis::common ALIAS jarvis_common)

target_include_directories(jarvis_common INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")

# The version is injected from the project() call so the header never drifts
# from the CMake project version.
target_compile_definitions(jarvis_common INTERFACE
    JARVIS_NATIVE_VERSION="${PROJECT_VERSION}")
