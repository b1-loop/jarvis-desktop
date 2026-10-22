# Local inference: thin wrappers over whisper.cpp (speech-to-text) and
# llama.cpp (text generation).
#
# Both projects are git submodules under native/third_party and ship their
# own copy of ggml. Each of them only adds its ggml copy when no `ggml`
# target exists yet ("otherwise assume ggml is added by a parent
# CMakeLists.txt"), so llama.cpp is added first and whisper.cpp then links
# against llama.cpp's ggml. One ggml, one CPU backend, no duplicate symbols.
#
# Third-party code is built as static libraries with their own warning
# settings (never /WX) and consumed as SYSTEM includes so their headers are
# exempt from our warning policy. CPU only for now; GPU backends are a
# follow-up (see docs/adr/0010-local-inference-wrappers.md).

set(JARVIS_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party")
if(NOT EXISTS "${JARVIS_THIRD_PARTY_DIR}/llama.cpp/CMakeLists.txt"
   OR NOT EXISTS "${JARVIS_THIRD_PARTY_DIR}/whisper.cpp/CMakeLists.txt")
    message(FATAL_ERROR
        "whisper.cpp / llama.cpp submodules are missing. Run "
        "'git submodule update --init --recursive' or configure with "
        "-DJARVIS_BUILD_INFERENCE=OFF.")
endif()

# --- Options for the submodules -------------------------------------------
# Plain set() configures the submodules once their option() calls honour
# normal variables (policy CMP0077). llama.cpp requires CMake 3.14 and gets
# that behaviour on its own; whisper.cpp still declares 3.5, under which
# option() would overwrite our values with cache defaults (and had turned
# BUILD_SHARED_LIBS on for the whole tree, Catch2 included).
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(BUILD_SHARED_LIBS OFF)
# llama.cpp is C++17 code: under C++20 its u8"" literals become char8_t and
# no longer stream into std::ostream. The standard is a directory-scoped
# variable, so it is lowered for the submodules and restored for our code.
set(CMAKE_CXX_STANDARD 17)
# A portable x86-64 build (SSE4.2/AVX/AVX2/FMA/F16C) instead of -march=native:
# the resulting .pyd is copied between machines and CI runners.
set(GGML_NATIVE OFF)
set(GGML_CCACHE OFF)
set(GGML_OPENMP ON)
# Nothing but the two libraries: no examples, tools, tests, server, or
# networking helpers, which would otherwise add minutes to the build.
set(LLAMA_BUILD_COMMON OFF)
set(LLAMA_BUILD_TESTS OFF)
set(LLAMA_BUILD_TOOLS OFF)
set(LLAMA_BUILD_EXAMPLES OFF)
set(LLAMA_BUILD_SERVER OFF)
set(LLAMA_BUILD_APP OFF)
set(LLAMA_OPENSSL OFF)
set(WHISPER_BUILD_TESTS OFF)
set(WHISPER_BUILD_EXAMPLES OFF)
set(WHISPER_BUILD_SERVER OFF)

add_subdirectory("${JARVIS_THIRD_PARTY_DIR}/llama.cpp" llama.cpp EXCLUDE_FROM_ALL SYSTEM)
add_subdirectory("${JARVIS_THIRD_PARTY_DIR}/whisper.cpp" whisper.cpp EXCLUDE_FROM_ALL SYSTEM)

if(WIN32)
    # whisper.cpp applies this to its own ggml copy: it disables the constexpr
    # std::mutex constructor so a process whose msvcp140.dll is older than the
    # toolset that built ggml does not crash in mutex code. We build ggml via
    # llama.cpp, so the define has to be repeated here.
    target_compile_definitions(ggml-base PRIVATE _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

set(CMAKE_CXX_STANDARD 20)

# --- The wrappers ----------------------------------------------------------
add_library(jarvis_inference_lib STATIC
    src/backend.cpp
    src/chat_template.cpp
    src/generation_stream.cpp
    src/llama_generator.cpp
    src/stop_matcher.cpp
    src/whisper_transcriber.cpp)
add_library(jarvis::inference ALIAS jarvis_inference_lib)

target_include_directories(jarvis_inference_lib PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(jarvis_inference_lib
    PUBLIC jarvis::common
    PRIVATE llama whisper)
jarvis_set_warnings(jarvis_inference_lib)
