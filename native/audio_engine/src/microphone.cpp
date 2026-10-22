#include "jarvis/audio/microphone.hpp"

#include "jarvis/common/errors.hpp"

#include <miniaudio.h>

#include <atomic>
#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace jarvis::audio {

namespace {

/// The notifications worth telling the owner about; "started" is the owner's
/// own doing and "unlocked" only exists on the Web Audio backend.
std::optional<DeviceNotification> toDeviceNotification(ma_device_notification_type type) noexcept {
    switch (type) {
        case ma_device_notification_type_stopped:
            return DeviceNotification::Stopped;
        case ma_device_notification_type_rerouted:
            return DeviceNotification::Rerouted;
        case ma_device_notification_type_interruption_began:
            return DeviceNotification::InterruptionBegan;
        case ma_device_notification_type_interruption_ended:
            return DeviceNotification::InterruptionEnded;
        case ma_device_notification_type_started:
        case ma_device_notification_type_unlocked:
            return std::nullopt;
    }
    return std::nullopt;
}

}  // namespace

struct MiniaudioMicrophone::Impl {
    // Zero-initialised for the pointer fields; ma_device_init overwrites the
    // rest, including the enum clang-tidy objects to having no zero value.
    ma_device device{};  // NOLINT(bugprone-invalid-enum-default-initialization)
    bool initialised{false};
    /// Set while stop() closes the device, so the "stopped" notification
    /// that closing fires is not mistaken for the device failing.
    std::atomic<bool> closing{false};
    AudioSource::FrameCallback onFrames;
    AudioSource::DeviceCallback onDevice;

    static void dataCallback(ma_device* device, void* /*output*/, const void* input,
                             ma_uint32 frameCount) {
        const auto* self = static_cast<const Impl*>(device->pUserData);
        if (self->onFrames && input != nullptr) {
            self->onFrames(std::span<const float>{static_cast<const float*>(input), frameCount});
        }
    }

    static void notificationCallback(const ma_device_notification* notification) {
        const auto* self = static_cast<const Impl*>(notification->pDevice->pUserData);
        if (self->closing.load() || !self->onDevice) {
            return;
        }
        if (const auto mapped = toDeviceNotification(notification->type)) {
            self->onDevice(*mapped);
        }
    }
};

MiniaudioMicrophone::MiniaudioMicrophone(int sampleRate)
    : impl_(std::make_unique<Impl>()), sampleRate_(sampleRate) {}

MiniaudioMicrophone::~MiniaudioMicrophone() {
    stop();
}

void MiniaudioMicrophone::start(FrameCallback onFrames, DeviceCallback onDevice) {
    if (impl_->initialised) {
        throw AudioDeviceError("microphone is already started");
    }
    impl_->onFrames = std::move(onFrames);
    impl_->onDevice = std::move(onDevice);
    impl_->closing.store(false);

    ma_device_config config = ma_device_config_init(ma_device_type_capture);
    config.capture.format = ma_format_f32;
    config.capture.channels = 1;
    config.sampleRate = static_cast<ma_uint32>(sampleRate_);
    config.dataCallback = &Impl::dataCallback;
    config.notificationCallback = &Impl::notificationCallback;
    config.pUserData = impl_.get();

    // A null device id opens the current default input, which is what makes
    // a restart after a device change land on the new default.
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
    impl_->closing.store(true);
    ma_device_uninit(&impl_->device);
    impl_->initialised = false;
    impl_->onFrames = nullptr;
    impl_->onDevice = nullptr;
}

}  // namespace jarvis::audio

// Design notes:
// - The notification callback only forwards; it never restarts the device.
//   miniaudio documents that ma_device_uninit/ma_device_start must not be
//   called from the device's own callbacks (they wait for the very thread
//   that is calling them), so recovery runs on the owner's thread instead.
// - A "stopped" notification can also precede miniaudio's own reroute to a
//   new default device (it stops, reroutes, restarts). The owner therefore
//   treats Stopped as "check that frames resume", not as "restart now".
