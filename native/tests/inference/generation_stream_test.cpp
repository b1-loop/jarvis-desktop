#include "jarvis/inference/generation_stream.hpp"

#include "jarvis/common/cancellation_token.hpp"
#include "jarvis/common/errors.hpp"
#include "jarvis/inference/llama_generator.hpp"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

using jarvis::CancellationToken;
using jarvis::inference::FinishReason;
using jarvis::inference::GenerationResult;
using jarvis::inference::GenerationStream;
using jarvis::inference::TokenCallback;

namespace {

/// A producer that emits fixed pieces, honouring cancellation like the real
/// generator does.
GenerationStream::Producer scripted(std::vector<std::string> pieces) {
    return
        [pieces = std::move(pieces)](const TokenCallback& emit, const CancellationToken& cancel) {
            GenerationResult result;
            for (const auto& piece : pieces) {
                if (cancel.cancelled() || !emit(piece)) {
                    result.finishReason = FinishReason::Cancelled;
                    return result;
                }
                result.text += piece;
                ++result.generatedTokens;
            }
            return result;
        };
}

std::string drain(GenerationStream& stream) {
    std::string text;
    while (const auto piece = stream.next()) {
        text += *piece;
    }
    return text;
}

}  // namespace

TEST_CASE("pieces arrive in order and the stream ends with nullopt", "[inference][stream]") {
    GenerationStream stream(scripted({"Hel", "lo", "!"}));

    REQUIRE(drain(stream) == "Hello!");
    REQUIRE_FALSE(stream.next().has_value());  // stays at the end
    REQUIRE(stream.finished());
    // value_or rather than ->: REQUIRE is a macro, so clang-tidy cannot see
    // the has_value() check that precedes a dereference.
    REQUIRE(stream.result().value_or(GenerationResult{}).generatedTokens == 3);
}

TEST_CASE("an exception in the producer is rethrown from next", "[inference][stream]") {
    GenerationStream stream([](const TokenCallback& emit, const CancellationToken&) {
        emit("partial");
        throw jarvis::ModelError("decode failed");
        return GenerationResult{};
    });

    REQUIRE(stream.next() == std::optional<std::string>{"partial"});
    REQUIRE_THROWS_AS(stream.next(), jarvis::ModelError);
    REQUIRE_FALSE(stream.result().has_value());
}

TEST_CASE("cancel stops a slow producer early", "[inference][stream]") {
    GenerationStream stream([](const TokenCallback& emit, const CancellationToken& cancel) {
        GenerationResult result;
        while (!cancel.cancelled()) {
            emit("x");
            ++result.generatedTokens;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        result.finishReason = FinishReason::Cancelled;
        return result;
    });

    REQUIRE(stream.next().has_value());
    stream.cancel();
    drain(stream);

    REQUIRE(stream.finished());
    REQUIRE(stream.result().value_or(GenerationResult{}).finishReason == FinishReason::Cancelled);
}

TEST_CASE("destroying an undrained stream joins the producer", "[inference][stream]") {
    bool stopped = false;
    {
        GenerationStream stream(
            [&stopped](const TokenCallback& emit, const CancellationToken& cancel) {
                while (!cancel.cancelled()) {
                    emit("x");
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                stopped = true;
                return GenerationResult{};
            });
        REQUIRE(stream.next().has_value());
    }
    REQUIRE(stopped);
}
