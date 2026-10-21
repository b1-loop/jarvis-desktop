#include "jarvis/audio/speaker.hpp"

#include "jarvis/common/errors.hpp"

#include <miniaudio.h>

#include <memory>
#include <span>
#include <utility>

namespace jarvis::audio {

struct MiniaudioSpeaker::Impl {
    // Zero-initialised for the pointer fields; ma_device_init overwrites the
    // rest, including the enum clang-tidy objects to having no zero value.
    ma_device device{};  // NOLINT(bugprone-invalid-enum-default-initialization)
    bool initialised{false};
    AudioSink::RenderCallback render;

    static void dataCallback(ma_device* device, void* output, const void* /*input*/,
                             ma_uint32 frameCount) {
        const auto* self = static_cast<const Impl*>(device->pUserData);
        if (self->render && output != nullptr) {
            self->render(std::span<float>{static_cast<float*>(output), frameCount});
        }
    }
};

MiniaudioSpeaker::MiniaudioSpeaker(int sampleRate)
    : impl_(std::make_unique<Impl>()), sampleRate_(sampleRate) {}

MiniaudioSpeaker::~MiniaudioSpeaker() {
    stop();
}

void MiniaudioSpeaker::start(RenderCallback render) {
    if (impl_->initialised) {
        throw AudioDeviceError("speaker is already started");
    }
    impl_->render = std::move(render);

    ma_device_config config = ma_device_config_init(ma_device_type_playback);
    config.playback.format = ma_format_f32;
    config.playback.channels = 1;
    config.sampleRate = static_cast<ma_uint32>(sampleRate_);
    config.dataCallback = &Impl::dataCallback;
    config.pUserData = impl_.get();

    if (ma_device_init(nullptr, &config, &impl_->device) != MA_SUCCESS) {
        throw AudioDeviceError("failed to open the default playback device");
    }
    impl_->initialised = true;
    if (ma_device_start(&impl_->device) != MA_SUCCESS) {
        stop();
        throw AudioDeviceError("failed to start the playback device");
    }
}

void MiniaudioSpeaker::stop() {
    if (!impl_->initialised) {
        return;
    }
    ma_device_uninit(&impl_->device);
    impl_->initialised = false;
    impl_->render = nullptr;
}

}  // namespace jarvis::audio
