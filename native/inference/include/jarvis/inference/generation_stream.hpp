/// @file generation_stream.hpp
/// @brief Pull-based hand-off of generated text from a native thread.

#pragma once

#include "jarvis/common/cancellation_token.hpp"
#include "jarvis/inference/llama_generator.hpp"

#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace jarvis::inference {

/// Runs one generation on a background thread and lets the consumer pull
/// pieces of text as they arrive.
///
/// This is the shape the Python binding needs: the producer never calls
/// into Python, so it never needs the GIL, and the consumer blocks in
/// next() with the GIL released. Errors thrown by the producer are stored
/// and rethrown from next(), so nothing is lost across the thread boundary.
class GenerationStream {
public:
    /// The work to run. It receives the callback that feeds the stream and
    /// the token that cancel() raises, and returns the final summary.
    using Producer =
        std::function<GenerationResult(const TokenCallback&, const CancellationToken&)>;

    /// Start @p producer on a new thread immediately.
    explicit GenerationStream(Producer producer);

    /// Cancels and joins; an abandoned stream never leaks a thread.
    ~GenerationStream();

    GenerationStream(const GenerationStream&) = delete;
    GenerationStream& operator=(const GenerationStream&) = delete;
    GenerationStream(GenerationStream&&) = delete;
    GenerationStream& operator=(GenerationStream&&) = delete;

    /// Block until the next piece of text is available. Returns std::nullopt
    /// once the producer has finished and every piece has been consumed.
    /// Rethrows any exception the producer raised.
    [[nodiscard]] std::optional<std::string> next();

    /// Ask the producer to stop. Pieces already produced remain readable.
    void cancel() noexcept;

    /// Whether the producer has returned (or thrown).
    [[nodiscard]] bool finished() const;

    /// The producer's summary, once finished() is true and no error occurred.
    [[nodiscard]] std::optional<GenerationResult> result() const;

private:
    void run(const Producer& producer);
    bool push(std::string_view piece);

    mutable std::mutex mutex_;
    std::condition_variable available_;
    std::deque<std::string> pieces_;
    std::optional<GenerationResult> result_;
    std::exception_ptr error_;
    bool finished_{false};
    CancellationToken cancel_;
    std::thread worker_;
};

}  // namespace jarvis::inference
