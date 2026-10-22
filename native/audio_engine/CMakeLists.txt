# Audio engine: capture, ring buffer, VAD, wake word, playback, barge-in.
#
# Third-party code is fetched at configure time and pinned by hash:
#  - miniaudio (single header) for device I/O,
#  - ONNX Runtime (prebuilt) for the Silero VAD and openWakeWord models.
# Both are consumed as SYSTEM includes so their warnings never hit /WX.

include(FetchContent)

if(NOT WIN32)
    message(FATAL_ERROR "The audio engine currently fetches the Windows ONNX Runtime build; "
                        "add the matching release URL for this platform in audio_engine/CMakeLists.txt.")
endif()

FetchContent_Declare(miniaudio
    URL https://raw.githubusercontent.com/mackron/miniaudio/0.11.22/miniaudio.h
    URL_HASH SHA256=9019743287e443c55e5737a7297f38e5e358561701d6db2d905afb114390c410
    DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_MakeAvailable(miniaudio)

FetchContent_Declare(onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-1.22.0.zip
    URL_HASH SHA256=174c616efc0271194488642a72f1a514e01487da4dfe84c49296d66e40ebe0da)
FetchContent_MakeAvailable(onnxruntime)

# --- miniaudio: compiled once, with the library's own warning level -------
add_library(jarvis_miniaudio STATIC src/miniaudio_impl.cpp)
target_include_directories(jarvis_miniaudio SYSTEM PUBLIC "${miniaudio_SOURCE_DIR}")
# We need device I/O and the MP3 decoder (edge-tts speech arrives as MP3);
# dropping the other codecs, the generators and the engine graph keeps the
# compile time of the implementation unit down.
target_compile_definitions(jarvis_miniaudio PUBLIC
    MA_NO_ENCODING
    MA_NO_WAV
    MA_NO_FLAC
    MA_NO_GENERATION
    MA_NO_RESOURCE_MANAGER
    MA_NO_NODE_GRAPH
    MA_NO_ENGINE)

# --- ONNX Runtime: imported prebuilt shared library ------------------------
add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.dll"
    IMPORTED_IMPLIB "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include"
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${onnxruntime_SOURCE_DIR}/include")

# Copies onnxruntime.dll next to a target's output so it can load at runtime.
function(jarvis_copy_onnxruntime_dll target)
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:onnxruntime::onnxruntime>"
            "$<TARGET_FILE_DIR:${target}>"
        COMMENT "Copying onnxruntime.dll next to ${target}")
endfunction()

# --- The engine ------------------------------------------------------------
add_library(jarvis_audio_engine STATIC
    src/engine.cpp
    src/microphone.cpp
    src/mp3_decoder.cpp
    src/onnx_session.cpp
    src/open_wake_word.cpp
    src/pipeline.cpp
    src/playback.cpp
    src/segmenter.cpp
    src/silero_vad.cpp
    src/speaker.cpp)
add_library(jarvis::audio_engine ALIAS jarvis_audio_engine)

target_include_directories(jarvis_audio_engine PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(jarvis_audio_engine
    PUBLIC jarvis::common onnxruntime::onnxruntime
    PRIVATE jarvis_miniaudio)
jarvis_set_warnings(jarvis_audio_engine)
