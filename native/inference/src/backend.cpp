#include "jarvis/inference/backend.hpp"

#include <ggml.h>
#include <llama.h>
#include <whisper.h>

#include <cstdio>
#include <mutex>
#include <string>
#include <tuple>

namespace jarvis::inference {

namespace {

/// Forwards warnings and errors to stderr and drops everything quieter.
///
/// ggml emits multi-line messages as one WARN/ERROR entry followed by CONT
/// entries, so the decision for the first line is remembered for the rest.
void filteredLog(ggml_log_level level, const char* text, void* /*user_data*/) {
    static thread_local bool forwarding = false;
    if (level != GGML_LOG_LEVEL_CONT) {
        forwarding = level >= GGML_LOG_LEVEL_WARN;
    }
    if (forwarding) {
        std::ignore = std::fputs(text, stderr);
    }
}

void initialiseOnce() {
    // whisper_log_set and llama_log_set each also reset ggml's callback, so
    // all three are set to the same filter and the order does not matter.
    llama_log_set(filteredLog, nullptr);
    whisper_log_set(filteredLog, nullptr);
    ggml_log_set(filteredLog, nullptr);
    llama_backend_init();
}

}  // namespace

void ensureBackend() {
    static std::once_flag once;
    std::call_once(once, initialiseOnce);
}

BackendVersions backendVersions() {
    return BackendVersions{.whisper = whisper_version(), .llama = llama_version()};
}

}  // namespace jarvis::inference

// Design notes:
// - llama_backend_free() is never called: the backend lives as long as the
//   process, exactly like the shared Ort::Env in the audio engine, and
//   tearing it down while a Python interpreter is unloading modules is a
//   known source of shutdown crashes.
// - Logging goes to stderr rather than through a callback into Python so
//   that a native thread never needs the GIL (ARCHITECTURE.md section 5).
