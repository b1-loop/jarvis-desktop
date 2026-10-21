# Catch2 unit tests for every native component. One executable, discovered
# test by test by ctest so failures are reported individually.

include(FetchContent)
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE
    SYSTEM)
FetchContent_MakeAvailable(Catch2)

add_executable(jarvis_tests
    version_test.cpp
    audio/ring_buffer_test.cpp
    audio/segmenter_test.cpp
    audio/playback_test.cpp
    audio/pipeline_test.cpp
    audio/engine_test.cpp
    audio/silero_vad_test.cpp
    audio/open_wake_word_test.cpp)
if(JARVIS_BUILD_INFERENCE)
    target_sources(jarvis_tests PRIVATE
        inference/cancellation_token_test.cpp
        inference/chat_template_test.cpp
        inference/generation_stream_test.cpp
        inference/llama_generator_test.cpp
        inference/stop_matcher_test.cpp
        inference/whisper_transcriber_test.cpp)
    target_link_libraries(jarvis_tests PRIVATE jarvis::inference)
endif()

target_include_directories(jarvis_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(jarvis_tests PRIVATE
    jarvis::common
    jarvis::audio_engine
    Catch2::Catch2WithMain)
# Model-backed tests skip themselves when the files are absent, so a checkout
# without downloaded models still runs the rest of the suite.
target_compile_definitions(jarvis_tests PRIVATE
    JARVIS_MODELS_DIR="${JARVIS_MODELS_DIR}")
jarvis_set_warnings(jarvis_tests)
jarvis_copy_onnxruntime_dll(jarvis_tests)

list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
include(Catch)
catch_discover_tests(jarvis_tests)
