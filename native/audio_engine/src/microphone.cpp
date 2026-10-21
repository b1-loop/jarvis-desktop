#include "jarvis/audio/microphone.hpp"

#include "jarvis/common/errors.hpp"

#include <miniaudio.h>

#include <memory>
#include <span>
#include <utility>

namespace jarvis::audio {

struct MiniaudioMicrophone::Impl {
    // Zero-initialised for the pointer fields; ma_device_init overwrites the
    // rest, including the enum clang-tidy objects to having no zero value.
    ma_device device{};  // NOLINT(bugprone-invalid-enum-default-initialization)
    bool initialised{false};
    AudioSource::FrameCallback onFrames;

    static void dataCallback(ma_device* device, void* /*output*/, const void* input,
                             ma_uint32 frameCount) {
        const auto* self = static_cast<const Impl*>(device->pUserData);
        if (self->onFrames && input != nullptr) {
            self->onFrames(std::span<const float>{static_cast<const float*>(input), frameCount});
        }
    }
};

MiniaudioMicrophone::MiniaudioMicrophone(int sampleRate)
    : impl_(std::make_unique<Impl>()), sampleRate_(sampleRate) {}

MiniaudioMicrophone::~MiniaudioMicrophone() {
    stop();
}

void MiniaudioMicrophone::start(FrameCallback onFrames) {
    if (impl_->initialised) {
        throw AudioDeviceError("microphone is already started");
    }
    impl_->onFrames = std::move(onFrames);

    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format = ma_format_f32;
    config.capture.channels = 1;
    config.sampleRate = static_cast<ma_uint32>(sampleRate_);
    config.dataCallback = &Impl::dataCallback;
    config.pUserData = impl_.get();

    if (ma_device_init(nullptr, &config, &impl_->device) != MA_SUCCESS) {
        throw AudioDeviceError("failed to open the default capture device");
    }
    impl_->initialised = true;
    if (ma_device_start(&impl_->device) != MA_SUCCESS) {
        stop();
        throw AudioDeviceError("failed to start the capture device");
    }
}

void MiniaudioMicrophone::stop() {
    if (!impl_->initialised) {
        return;
    }
    ma_device_uninit(&impl_->device);
    impl_->initialised = false;
    impl_->onFrames = nullptr;
}

}  // namespace jarvis::audio
