#include "jarvis/inference/generation_stream.hpp"

#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace jarvis::inference {

GenerationStream::GenerationStream(Producer producer)
    : worker_([this, producer = std::move(producer)] { run(producer); }) {}

GenerationStream::~GenerationStream() {
    cancel();
    if (worker_.joinable()) {
        worker_.join();
    }
}

std::optional<std::string> GenerationStream::next() {
    std::unique_lock lock(mutex_);
    available_.wait(lock, [this] { return !pieces_.empty() || finished_; });
    if (!pieces_.empty()) {
        std::string piece = std::move(pieces_.front());
        pieces_.pop_front();
        return piece;
    }
    if (error_ != nullptr) {
        std::rethrow_exception(std::exchange(error_, nullptr));
    }
    return std::nullopt;
}

void GenerationStream::cancel() noexcept {
    cancel_.cancel();
}

bool GenerationStream::finished() const {
    const std::scoped_lock lock(mutex_);
    return finished_;
}

std::optional<GenerationResult> GenerationStream::result() const {
    const std::scoped_lock lock(mutex_);
    return result_;
}

void GenerationStream::run(const Producer& producer) {
    std::optional<GenerationResult> result;
    std::exception_ptr error;
    try {
        result = producer([this](std::string_view piece) { return push(piece); }, cancel_);
    } catch (...) {
        error = std::current_exception();
    }
    const std::scoped_lock lock(mutex_);
    result_ = std::move(result);
    error_ = error;
    finished_ = true;
    available_.notify_all();
}

bool GenerationStream::push(std::string_view piece) {
    {
        const std::scoped_lock lock(mutex_);
        pieces_.emplace_back(piece);
    }
    available_.notify_one();
    return !cancel_.cancelled();
}

}  // namespace jarvis::inference

// Design notes:
// - The producer is injected as a std::function so the stream is tested
//   with a scripted producer and no model; LlamaGenerator supplies the real
//   one in the Python binding.
// - The queue is unbounded on purpose: a CPU model produces tens of tokens
//   per second and each piece is a few bytes, while bounding it would make
//   the producer block on the consumer and complicate cancellation.
// - An error is rethrown exactly once, from the next() call that observes
//   the end of the stream, so the Python caller sees it where it reads.
