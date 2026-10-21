#include "jarvis/common/version.hpp"

#include <catch2/catch_test_macros.hpp>

#include <regex>
#include <string>

TEST_CASE("native version is a semantic version", "[common]") {
    const std::regex semanticVersion{R"(^\d+\.\d+\.\d+$)"};
    REQUIRE(std::regex_match(std::string{jarvis::common::kVersion}, semanticVersion));
}
