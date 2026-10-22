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
    audio/replay_source_test.cpp
    audio/mp3_decoder_test.cpp
    audio/silero_vad_test.cpp
    audio/open_wake_word_test.cpp
    os/audio_endpoint_test.cpp
    os/clipboard_test.cpp
    os/hotkey_manager_test.cpp
    os/hotkey_parser_test.cpp
    os/process_control_test.cpp
    os/screenshot_test.cpp
    os/system_controls_test.cpp
    os/text_encoding_test.cpp
    os/window_manager_test.cpp)
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
    jarvis::os_integration
    wtsapi32
    Catch2::Catch2WithMain)
# Model-backed tests skip themselves when the files are absent, so a checkout
# without downloaded models still runs the rest of the suite. Fixtures are
# shared with the Python suite so both decode the same clip. The OS tests
# include <windows.h> for their desktop probe, hence the lean/no-min-max defines.
target_compile_definitions(jarvis_tests PRIVATE
    JARVIS_MODELS_DIR="${JARVIS_MODELS_DIR}"
    JARVIS_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fixtures"
    WIN32_LEAN_AND_MEAN
    NOMINMAX)
jarvis_set_warnings(jarvis_tests)
jarvis_copy_onnxruntime_dll(jarvis_tests)

if(JARVIS_BUILD_OVERLAY)
    target_sources(jarvis_tests PRIVATE
        overlay/animation_test.cpp
        overlay/boot_test.cpp
        overlay/effects_test.cpp
        overlay/hud_data_test.cpp
        overlay/hud_layout_test.cpp
        overlay/interactive_window_test.cpp
        overlay/layout_test.cpp
        overlay/minimize_policy_test.cpp
        overlay/minimize_window_test.cpp
        overlay/model_test.cpp
        overlay/orb_test.cpp
        overlay/particles_test.cpp
        overlay/placement_test.cpp
        overlay/scene_test.cpp
        overlay/shaders_test.cpp
        overlay/text_entry_test.cpp
        overlay/ui_events_test.cpp
        overlay/window_test.cpp)
    # text_entry_test drives the private TextEntry over a headless ImGui context.
    target_include_directories(jarvis_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../overlay/src")
    target_link_libraries(jarvis_tests PRIVATE jarvis::overlay jarvis_imgui)
endif()

list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
include(Catch)
catch_discover_tests(jarvis_tests)
