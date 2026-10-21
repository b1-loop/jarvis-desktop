/// @file process_control.hpp
/// @brief Launch applications and close them gracefully or by force.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace jarvis::os {

/// Interface for starting and stopping processes.
///
/// "Close" asks the process's top-level windows to close (WM_CLOSE) so the
/// application can save state or ask the user; "force" terminates it
/// outright. The two are separate flags rather than a fallback because
/// killing an editor with unsaved work is exactly the kind of action the
/// agent must confirm first, not slide into.
class ProcessControl {
public:
    ProcessControl() = default;
    ProcessControl(const ProcessControl&) = delete;
    ProcessControl& operator=(const ProcessControl&) = delete;
    ProcessControl(ProcessControl&&) = delete;
    ProcessControl& operator=(ProcessControl&&) = delete;
    virtual ~ProcessControl() = default;

    /// Start @p target with @p arguments, where target is an executable
    /// path, a name on PATH, a document or a URL (anything the shell can
    /// "open"). Returns the new process id, or 0 when the shell handed the
    /// target to an already-running process (typical for documents).
    virtual std::uint32_t launch(std::string_view target, std::string_view arguments) = 0;

    /// Close the process with @p pid. With @p force false, every top-level
    /// window of the process receives WM_CLOSE and an OsError is raised if
    /// it has none; with force true the process is terminated.
    /// Throws ProcessNotFoundError if no such process is running.
    virtual void close(std::uint32_t pid, bool force) = 0;

    /// Apply close() to every process whose executable name matches
    /// @p executableName (case-insensitive, ".exe" optional) and return how
    /// many were addressed. Throws ProcessNotFoundError if none matched.
    virtual std::size_t closeByName(std::string_view executableName, bool force) = 0;
};

/// Win32 implementation over ShellExecuteEx, EnumWindows and TerminateProcess.
class Win32ProcessControl final : public ProcessControl {
public:
    std::uint32_t launch(std::string_view target, std::string_view arguments) override;
    void close(std::uint32_t pid, bool force) override;
    std::size_t closeByName(std::string_view executableName, bool force) override;
};

}  // namespace jarvis::os
