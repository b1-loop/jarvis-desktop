/// @file notifier.hpp
/// @brief System notifications.

#pragma once

#include <memory>
#include <string_view>

namespace jarvis::os {

/// Interface for showing a notification to the user.
class Notifier {
public:
    Notifier() = default;
    Notifier(const Notifier&) = delete;
    Notifier& operator=(const Notifier&) = delete;
    Notifier(Notifier&&) = delete;
    Notifier& operator=(Notifier&&) = delete;
    virtual ~Notifier() = default;

    /// Show @p message under @p title. Returns once the request is queued,
    /// not when the notification disappears.
    virtual void notify(std::string_view title, std::string_view message) = 0;
};

/// Notifications through a notification-area icon and Shell_NotifyIcon's
/// NIF_INFO "balloon", which Windows 10 and 11 render as a regular toast
/// in the Action Center.
///
/// Limits of this approach (accepted, see docs/adr/0004): the title is cut
/// at 63 characters and the body at 255, there are no action buttons, and
/// an icon sits in the notification area for the lifetime of the object.
/// In exchange it needs no app registration, no WinRT and no package
/// identity, so it works from a plain Python process.
class BalloonNotifier final : public Notifier {
public:
    /// Create the hidden window and the tray icon. Throws OsError on failure.
    explicit BalloonNotifier(std::string_view tooltip = "Jarvis");
    ~BalloonNotifier() override;

    BalloonNotifier(const BalloonNotifier&) = delete;
    BalloonNotifier& operator=(const BalloonNotifier&) = delete;
    BalloonNotifier(BalloonNotifier&&) = delete;
    BalloonNotifier& operator=(BalloonNotifier&&) = delete;

    void notify(std::string_view title, std::string_view message) override;

private:
    struct Impl;  // owns the message loop thread and the hidden window
    std::unique_ptr<Impl> impl_;
};

}  // namespace jarvis::os
