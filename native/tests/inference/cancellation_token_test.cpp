#include "jarvis/common/cancellation_token.hpp"

#include <catch2/catch_test_macros.hpp>

#include <thread>

using jarvis::CancellationToken;

TEST_CASE("a fresh token is not cancelled", "[inference][cancellation]") {
    const CancellationToken token;
    REQUIRE_FALSE(token.cancelled());
}

TEST_CASE("cancel raises the flag and is idempotent", "[inference][cancellation]") {
    CancellationToken token;
    token.cancel();
    token.cancel();
    REQUIRE(token.cancelled());
}

TEST_CASE("reset lowers the flag for reuse", "[inference][cancellation]") {
    CancellationToken token;
    token.cancel();
    token.reset();
    REQUIRE_FALSE(token.cancelled());
}

TEST_CASE("cancellation from another thread is observed", "[inference][cancellation]") {
    CancellationToken token;
    std::thread canceller([&token] { token.cancel(); });
    canceller.join();
    REQUIRE(token.cancelled());
}
