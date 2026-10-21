/// @file backend.hpp
/// @brief Process-wide ggml/llama/whisper runtime setup.

#pragma once

#include <string>

namespace jarvis::inference {

/// Initialise the shared ggml backend once per process and route the
/// libraries' logging through Jarvis' filter.
///
/// Both wrappers call this from their constructors; repeated calls are
/// no-ops. whisper.cpp and llama.cpp log model metadata at INFO level on
/// every load, which would otherwise flood the terminal of a voice
/// assistant, so only warnings and errors are forwarded to stderr.
void ensureBackend();

/// Version strings of the pinned submodules, for diagnostics and the
/// Python module's metadata.
struct BackendVersions {
    std::string whisper;
    std::string llama;
};

[[nodiscard]] BackendVersions backendVersions();

}  // namespace jarvis::inference
