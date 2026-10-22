/// @file system_controls.hpp
/// @brief Master volume, display brightness and the lock screen.

#pragma once

namespace jarvis::os {

/// Interface for the handful of system knobs a voice assistant is asked for.
///
/// Levels are integers 0-100 because that is how people say them; the
/// implementation converts to whatever scale the API uses.
class SystemControls {
public:
    SystemControls() = default;
    SystemControls(const SystemControls&) = delete;
    SystemControls& operator=(const SystemControls&) = delete;
    SystemControls(SystemControls&&) = delete;
    SystemControls& operator=(SystemControls&&) = delete;
    virtual ~SystemControls() = default;

    /// Master volume of the default output device, 0-100.
    /// Throws OsError when there is no audio endpoint.
    [[nodiscard]] virtual int volume() const = 0;

    /// Set the master volume. Values outside 0-100 are clamped.
    virtual void setVolume(int percent) = 0;

    [[nodiscard]] virtual bool muted() const = 0;
    virtual void setMuted(bool muted) = 0;

    /// Brightness of the built-in display, 0-100.
    /// Throws UnsupportedError when no display exposes software brightness
    /// (desktops with external monitors; see the README for the scope).
    [[nodiscard]] virtual int brightness() const = 0;

    /// Set the built-in display brightness. Values outside 0-100 are clamped.
    virtual void setBrightness(int percent) = 0;

    /// Lock the workstation, as Win+L does.
    virtual void lockScreen() = 0;
};

/// Core Audio (IAudioEndpointVolume), WMI (WmiMonitorBrightnessMethods) and
/// LockWorkStation. COM is initialised per call with RAII so the calling
/// thread's apartment state is left exactly as it was found.
class Win32SystemControls final : public SystemControls {
public:
    [[nodiscard]] int volume() const override;
    void setVolume(int percent) override;
    [[nodiscard]] bool muted() const override;
    void setMuted(bool muted) override;
    [[nodiscard]] int brightness() const override;
    void setBrightness(int percent) override;
    void lockScreen() override;
};

}  // namespace jarvis::os
