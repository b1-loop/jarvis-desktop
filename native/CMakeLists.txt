# Root build for the Jarvis native layer.
#
# Produces four pybind11 modules (jarvis_audio, jarvis_os, jarvis_inference,
# jarvis_overlay) and a Catch2 test binary. Components are switched on as they
# land; until then their option defaults to OFF and the corresponding Python
# module exposes only version information. Invoke through
# scripts/build_native.py, which supplies the Python interpreter, the pybind11
# CMake directory and the output directory.

cmake_minimum_required(VERSION 3.28)

project(jarvis_native
    VERSION 0.1.0
    DESCRIPTION "Native hot paths for the Jarvis voice assistant"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(JARVIS_BUILD_TESTS "Build the Catch2 unit tests" ON)
option(JARVIS_BUILD_OS "Build the OS integration layer (lands in step 4)" OFF)
option(JARVIS_BUILD_INFERENCE "Build the whisper.cpp / llama.cpp wrappers (lands in step 12)" OFF)
option(JARVIS_BUILD_OVERLAY "Build the Dear ImGui overlay (lands in step 13)" OFF)

set(JARVIS_PYTHON_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../src/jarvis/_native"
    CACHE PATH "Directory that receives the built Python extension modules")
set(JARVIS_MODELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../models"
    CACHE PATH "Directory holding the ONNX models downloaded by scripts/download_models.py")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

add_subdirectory(common)
add_subdirectory(audio_engine)

if(JARVIS_BUILD_OS)
    add_subdirectory(os_integration)
endif()
if(JARVIS_BUILD_INFERENCE)
    add_subdirectory(inference)
endif()
if(JARVIS_BUILD_OVERLAY)
    add_subdirectory(overlay)
endif()

add_subdirectory(bindings)

if(JARVIS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
