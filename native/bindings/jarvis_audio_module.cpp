/// @file jarvis_audio_module.cpp
/// @brief Python module @c jarvis_audio.
///
/// A thin layer: converts numpy arrays to spans, releases the GIL around
/// calls that touch devices or locks, and maps NativeError onto Python
/// exceptions. All behaviour lives in jarvis::audio.

#include "jarvis/audio/audio_format.hpp"
#include "jarvis/audio/engine.hpp"
#include "jarvis/audio/events.hpp"
#include "jarvis/audio/microphone.hpp"
#include "jarvis/audio/mp3_decoder.hpp"
#include "jarvis/audio/open_wake_word.hpp"
#include "jarvis/audio/pipeline.hpp"
#include "jarvis/audio/playback.hpp"
#include "jarvis/audio/silero_vad.hpp"
#include "jarvis/audio/speaker.hpp"
#include "jarvis/audio/vad.hpp"
#include "jarvis/common/errors.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "module_info.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;
using namespace jarvis::audio;  // NOLINT(google-build-using-namespace) - binding file only

namespace {

// NOLINTNEXTLINE(bugprone-signed-bitwise) - pybind11's documented flag idiom
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using WakeWordModelPaths =
    std::tuple<std::filesystem::path, std::filesystem::path, std::filesystem::path>;

std::span<const float> asSpan(const FloatArray& array) {
    if (array.ndim() != 1) {
        throw py::value_error("expected a one-dimensional float32 array");
    }
    return {array.data(), static_cast<std::size_t>(array.shape(0))};
}

FloatArray toArray(const std::vector<float>& samples) {
    FloatArray array(static_cast<py::ssize_t>(samples.size()));
    std::ranges::copy(samples, array.mutable_data());
    return array;
}

std::unique_ptr<AudioEngine> makeEngine(const std::filesystem::path& vadModel,
                                        const std::optional<WakeWordModelPaths>& wakeWordModels,
                                        const EngineConfig& config) {
    std::unique_ptr<WakeWordDetector> wakeWord;
    if (wakeWordModels.has_value()) {
        const auto& [melspectrogram, embedding, classifier] = *wakeWordModels;
        wakeWord = std::make_unique<OpenWakeWord>(OpenWakeWord::ModelPaths{
            .melspectrogram = melspectrogram, .embedding = embedding, .classifier = classifier});
    }
    return std::make_unique<AudioEngine>(std::make_unique<MiniaudioMicrophone>(),
                                         std::make_unique<SileroVad>(vadModel), std::move(wakeWord),
                                         config);
}

// Translators are consulted most-recently-registered first, so the base
// class goes first and the derived classes after it.
void defineExceptions(const py::module_& module) {
    const auto& nativeError = py::register_local_exception<jarvis::NativeError>(
        module, "NativeError", PyExc_RuntimeError);
    py::register_local_exception<jarvis::AudioDeviceError>(module, "AudioDeviceError",
                                                           nativeError.ptr());
    py::register_local_exception<jarvis::ModelError>(module, "ModelError", nativeError.ptr());
    py::register_local_exception<jarvis::DecodeError>(module, "DecodeError", nativeError.ptr());
}

std::span<const std::byte> asBytes(const py::bytes& data) {
    char* buffer = nullptr;
    py::ssize_t length = 0;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(data.ptr(), &buffer, &length) != 0) {
        throw py::error_already_set();
    }
    return std::as_bytes(std::span<const char>{buffer, static_cast<std::size_t>(length)});
}

void defineDecoding(py::module_& module) {
    module.def(
        "decode_mp3",
        [](const py::bytes& data) {
            // The bytes object is immutable and stays referenced by `data`,
            // so its buffer can be read with the GIL released.
            const auto bytes = asBytes(data);
            DecodedAudio audio;
            {
                const py::gil_scoped_release release;
                audio = decodeMp3(bytes);
            }
            return py::make_tuple(toArray(toMono(audio)), audio.sampleRate);
        },
        py::arg("data"),
        "Decode a complete MP3 buffer to (mono float32 samples, sample_rate). "
        "Raises DecodeError if no frame could be decoded.");
}

void defineConfig(const py::module_& module) {
    py::class_<SegmenterConfig>(module, "SegmenterConfig",
                                "Thresholds for turning VAD probabilities into speech segments.")
        .def(py::init<>())
        .def_readwrite("start_threshold", &SegmenterConfig::startThreshold)
        .def_readwrite("end_threshold", &SegmenterConfig::endThreshold)
        .def_readwrite("min_speech_ms", &SegmenterConfig::minSpeechMs)
        .def_readwrite("silence_end_ms", &SegmenterConfig::silenceEndMs);

    py::class_<PipelineConfig>(module, "PipelineConfig", "Detection tuning for the engine.")
        .def(py::init<>())
        .def_readwrite("segmenter", &PipelineConfig::segmenter)
        .def_readwrite("wake_word_threshold", &PipelineConfig::wakeWordThreshold)
        .def_readwrite("wake_word_cooldown_ms", &PipelineConfig::wakeWordCooldownMs)
        .def_readwrite("pre_roll_ms", &PipelineConfig::preRollMs);

    py::class_<EngineConfig>(module, "EngineConfig", "Threading and buffering parameters.")
        .def(py::init<>())
        .def_readwrite("pipeline", &EngineConfig::pipeline)
        .def_readwrite("ring_buffer_samples", &EngineConfig::ringBufferSamples)
        .def_readwrite("worker_poll_ms", &EngineConfig::workerPollMs);
}

void defineEvents(const py::module_& module) {
    py::enum_<AudioEventType>(module, "AudioEventType")
        .value("WAKE_WORD_DETECTED", AudioEventType::WakeWordDetected)
        .value("SPEECH_STARTED", AudioEventType::SpeechStarted)
        .value("SPEECH_ENDED", AudioEventType::SpeechEnded)
        .value("SAMPLES_DROPPED", AudioEventType::SamplesDropped);

    py::class_<AudioEvent>(module, "AudioEvent", "One event reported by the engine.")
        .def_readonly("type", &AudioEvent::type)
        .def_readonly("value", &AudioEvent::value)
        .def_readonly("sample_position", &AudioEvent::samplePosition)
        .def("__repr__", [](const AudioEvent& event) {
            return "AudioEvent(type=" + std::string{py::str(py::cast(event.type))} +
                   ", value=" + std::to_string(event.value) +
                   ", sample_position=" + std::to_string(event.samplePosition) + ")";
        });
}

void definePlayback(const py::module_& module) {
    py::class_<Playback, std::shared_ptr<Playback>>(
        module, "Playback", "Streaming playback on the default output device with instant stop.")
        .def(py::init([](int sampleRate) {
                 return std::make_shared<Playback>(std::make_unique<MiniaudioSpeaker>(sampleRate));
             }),
             py::arg("sample_rate"), py::call_guard<py::gil_scoped_release>())
        .def(
            "enqueue",
            [](Playback& self, const FloatArray& samples) { self.enqueue(asSpan(samples)); },
            py::arg("samples"), "Queue a one-dimensional float32 array of mono samples.")
        .def("stop", &Playback::stop, "Discard everything queued and fall silent now.")
        .def("mark_end_of_stream", &Playback::markEndOfStream)
        .def("is_playing", &Playback::isPlaying)
        .def("queued_samples", &Playback::queuedSamples)
        .def("output_level", &Playback::outputLevel)
        .def("stop_count", &Playback::stopCount,
             "How many times stop() has run, including native barge-in stops.")
        .def("sample_rate", &Playback::sampleRate);
}

void defineEngine(const py::module_& module) {
    py::class_<AudioEngine>(module, "AudioEngine",
                            "Capture, VAD and wake-word detection on native threads.")
        .def(py::init(&makeEngine), py::arg("vad_model"), py::arg("wake_word_models") = py::none(),
             py::arg("config") = EngineConfig{},
             "Load the models. wake_word_models is (melspectrogram, embedding, classifier) "
             "or None for hotkey-only operation.")
        .def("start", &AudioEngine::start, py::call_guard<py::gil_scoped_release>())
        .def("stop", &AudioEngine::stop, py::call_guard<py::gil_scoped_release>())
        .def("running", &AudioEngine::running)
        .def("set_listening", &AudioEngine::setListening, py::arg("listening"))
        .def("listening", &AudioEngine::listening)
        .def(
            "take_captured", [](AudioEngine& self) { return toArray(self.takeCaptured()); },
            "Return and clear the audio captured while listening, as float32 at 16 kHz.")
        .def("poll_events", &AudioEngine::pollEvents)
        .def("set_barge_in_target", &AudioEngine::setBargeInTarget, py::arg("playback"))
        .def("input_level", &AudioEngine::inputLevel)
        .def("dropped_samples", &AudioEngine::droppedSamples)
        .def("samples_processed", &AudioEngine::samplesProcessed);
}

}  // namespace

PYBIND11_MODULE(jarvis_audio, module) {
    module.doc() =
        "Audio engine: microphone capture, ring buffer, Silero VAD, openWakeWord and "
        "playback with barge-in. See native/audio_engine/README.md.";
    jarvis::bindings::defineModuleInfo(module, "implemented in step 3");

    module.attr("SAMPLE_RATE") = kSampleRate;
    module.attr("VAD_FRAME_SAMPLES") = kVadFrameSamples;
    module.attr("WAKE_WORD_CHUNK_SAMPLES") = kWakeWordChunkSamples;

    defineExceptions(module);
    defineConfig(module);
    defineEvents(module);
    definePlayback(module);
    defineEngine(module);
    defineDecoding(module);
}
