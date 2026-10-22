/// @file model.hpp
/// @brief Thread-safe state behind the overlay: what Python says, what the
/// UI thread draws.

#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace jarvis::overlay {

/// Mirror of the Python state machine (core/state_machine.py). The overlay
/// only displays it; transitions are validated on the Python side.
enum class AssistantState : std::uint8_t { Idle, Listening, Thinking, Speaking };

/// Upper-case member name as exposed to Python, e.g. "LISTENING".
[[nodiscard]] std::string_view toString(AssistantState state) noexcept;

/// Inverse of toString(); std::nullopt for anything else.
[[nodiscard]] std::optional<AssistantState> parseAssistantState(std::string_view name) noexcept;

/// Fixed-capacity history of level samples, oldest overwritten first.
///
/// Levels arrive from Python at the audio engine's polling cadence (about
/// every 10 ms) and the UI thread wants the last few hundred milliseconds
/// as bars, so a ring that never allocates after construction is the right
/// shape. Not thread-safe on its own; OverlayModel serialises access.
class LevelRing {
public:
    explicit LevelRing(std::size_t capacity);

    /// Append @p level (clamped to [0, 1]), evicting the oldest sample.
    void push(float level) noexcept;

    /// Reset every sample to silence.
    void clear() noexcept;

    /// Samples in chronological order, oldest first; always capacity() long.
    [[nodiscard]] std::vector<float> ordered() const;

    [[nodiscard]] std::size_t capacity() const noexcept { return levels_.size(); }
    [[nodiscard]] float latest() const noexcept;

private:
    std::vector<float> levels_;
    std::size_t next_{0};
};

/// One line of the boot sequence's diagnostic readout, e.g. "MODEL" /
/// "qwen2.5:7b". The values are real: Python passes what it is running.
struct ReadoutLine {
    std::string label;
    std::string value;

    [[nodiscard]] bool operator==(const ReadoutLine&) const noexcept = default;
};

/// The readout, in display order.
using BootReadout = std::vector<ReadoutLine>;

/// Immutable copy of everything the UI thread needs for one frame.
struct OverlaySnapshot {
    AssistantState state{AssistantState::Idle};
    /// Oldest to newest, each in [0, 1].
    std::vector<float> levels;
    /// What the user said (UTF-8, already bounded by the model).
    std::string transcript;
    /// What Jarvis is saying (UTF-8, already bounded by the model).
    std::string response;
    bool visible{true};
    /// 1 right after the wake word was heard, decaying linearly to 0.
    float wakeFlash{0.0F};
    /// When the boot sequence was requested; unset until playBoot().
    std::optional<std::chrono::steady_clock::time_point> bootStartedAt;
    /// When the boot sequence was skipped, if it was.
    std::optional<std::chrono::steady_clock::time_point> bootSkippedAt;
    /// Diagnostic lines shown during the boot's systems check; shared, not
    /// copied, because it never changes once the boot has started.
    std::shared_ptr<const BootReadout> bootReadout;
    /// Incremented by every mutation; lets the renderer slow down when idle.
    std::uint64_t revision{0};
};

/// Bounds that keep the model's memory use independent of what Python sends.
struct ModelLimits {
    /// Number of level samples kept for the waveform.
    std::size_t levelHistory = 64;
    /// Maximum characters (code points) kept per text line; the tail wins.
    std::size_t maxTextChars = 240;
    /// How long the wake-word flash stays visible.
    std::chrono::milliseconds wakeFlashDuration{600};
    /// Boot readout lines kept; later ones are dropped.
    std::size_t maxReadoutLines = 6;
    /// Characters kept per readout label and per value (the head wins).
    std::size_t maxReadoutLabelChars = 12;
    std::size_t maxReadoutValueChars = 34;
};

/// The overlay's state, owned by the window and mutated from Python.
///
/// Thread model: setters run on whichever thread Python calls them from and
/// only ever take the mutex for a few hundred nanoseconds; the UI thread
/// takes a snapshot() once per frame. Nothing here blocks on rendering, so a
/// slow GPU can never stall the event loop.
class OverlayModel {
public:
    using Clock = std::chrono::steady_clock;

    explicit OverlayModel(ModelLimits limits = {});

    /// Entering Idle also clears the level history so the waveform settles
    /// immediately instead of showing the last utterance's tail. Leaving
    /// Idle while the boot sequence plays skips it: the user is talking to
    /// Jarvis, so the panel should be in its corner, not mid-intro.
    void setState(AssistantState state, Clock::time_point now = Clock::now());
    void pushLevel(float level);
    void setTranscript(std::string_view utf8);
    void setResponse(std::string_view utf8);
    void setVisible(bool visible);
    void flashWakeWord(Clock::time_point now = Clock::now());

    /// Start the boot sequence at @p now. Idempotent: only the first call
    /// in the model's lifetime starts it.
    /// @return true if this call started it.
    bool playBoot(Clock::time_point now = Clock::now());

    /// Same, with a diagnostic @p readout for the systems check. Lines and
    /// characters beyond the limits are dropped. A repeat call changes
    /// nothing, including the readout.
    bool playBoot(BootReadout readout, Clock::time_point now = Clock::now());

    /// Skip the boot sequence at @p now (it still eases into the corner).
    /// @return true if a started, not yet skipped boot was skipped.
    bool skipBoot(Clock::time_point now = Clock::now());

    [[nodiscard]] OverlaySnapshot snapshot(Clock::time_point now = Clock::now()) const;
    [[nodiscard]] AssistantState state() const;
    [[nodiscard]] bool visible() const;

    /// Lock-free view of the mutation counter for cheap "anything new?" polls.
    [[nodiscard]] std::uint64_t revision() const noexcept {
        return revision_.load(std::memory_order_acquire);
    }

    [[nodiscard]] const ModelLimits& limits() const noexcept { return limits_; }

private:
    /// Caller holds mutex_.
    void touch() noexcept;
    /// @p readout cut to the limits; no lock needed.
    [[nodiscard]] BootReadout boundedReadout(BootReadout readout) const;
    [[nodiscard]] float wakeFlashAt(Clock::time_point now) const;

    ModelLimits limits_;
    mutable std::mutex mutex_;  // guards every field below except revision_
    AssistantState state_{AssistantState::Idle};
    LevelRing levels_;
    std::string transcript_;
    std::string response_;
    bool visible_{true};
    std::optional<Clock::time_point> wakeFlashAt_;
    std::optional<Clock::time_point> bootStartedAt_;
    std::optional<Clock::time_point> bootSkippedAt_;
    std::shared_ptr<const BootReadout> bootReadout_;
    std::atomic<std::uint64_t> revision_{0};
};

}  // namespace jarvis::overlay
