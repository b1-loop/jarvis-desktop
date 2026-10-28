#include "jarvis/os/clipboard.hpp"

#include "jarvis/os/errors.hpp"

#include <catch2/catch_test_macros.hpp>

#include "os/desktop.hpp"
#include <windows.h>

#include <chrono>
#include <string>

using jarvis::os::ClipboardBusyError;
using jarvis::os::ClipboardOpenPolicy;
using jarvis::os::OsError;
using jarvis::os::Win32Clipboard;
using jarvis::os::testing::requireDesktop;
using jarvis::os::testing::requireStateChangesAllowed;

namespace {

constexpr int kTestAttempts = 3;

/// The clipboard text, or SKIP when another application keeps the clipboard
/// open. Any other failure still fails the test.
std::string textOrSkip(const Win32Clipboard& clipboard) {
    try {
        return clipboard.text();
    } catch (const ClipboardBusyError& error) {
        SKIP("clipboard unavailable: " << error.what());
    }
    return {};
}

/// A policy whose first @p failures attempts fail with @p error and whose
/// later attempts open the real clipboard. Counts attempts in @p calls.
ClipboardOpenPolicy failing(int failures, DWORD error, int& calls) {
    return ClipboardOpenPolicy{.open =
                                   [failures, error, &calls] {
                                       ++calls;
                                       if (calls <= failures) {
                                           SetLastError(error);
                                           return false;
                                       }
                                       return OpenClipboard(nullptr) != FALSE;
                                   },
                               .attempts = kTestAttempts,
                               .firstDelay = std::chrono::milliseconds(1)};
}

}  // namespace

TEST_CASE("reading the clipboard never throws on a desktop", "[os][clipboard]") {
    requireDesktop();
    const Win32Clipboard clipboard;

    static_cast<void>(textOrSkip(clipboard));
}

TEST_CASE("a clipboard that stays denied is reported as busy", "[os][clipboard]") {
    int calls = 0;
    const Win32Clipboard clipboard{failing(kTestAttempts, ERROR_ACCESS_DENIED, calls)};

    REQUIRE_THROWS_AS(clipboard.text(), ClipboardBusyError);
    REQUIRE(calls == kTestAttempts);
}

TEST_CASE("other open failures stay generic OS errors", "[os][clipboard]") {
    int calls = 0;
    const Win32Clipboard clipboard{failing(kTestAttempts, ERROR_NOT_ENOUGH_MEMORY, calls)};

    bool busy = false;
    bool generic = false;
    try {
        static_cast<void>(clipboard.text());
    } catch (const ClipboardBusyError&) {
        busy = true;
    } catch (const OsError&) {
        generic = true;
    }
    REQUIRE_FALSE(busy);
    REQUIRE(generic);
}

TEST_CASE("a briefly denied clipboard is read after retrying", "[os][clipboard]") {
    requireDesktop();
    int calls = 0;
    const Win32Clipboard clipboard{failing(kTestAttempts - 1, ERROR_ACCESS_DENIED, calls)};

    static_cast<void>(textOrSkip(clipboard));
    REQUIRE(calls == kTestAttempts);
}

TEST_CASE("text survives a set/get round trip", "[os][clipboard][state]") {
    requireStateChangesAllowed();
    Win32Clipboard clipboard;
    const std::string original = textOrSkip(clipboard);
    const std::string sample = "Jarvis clipboard \xE2\x9C\x93 \xF0\x9F\x9A\x80\nline two";

    clipboard.setText(sample);
    REQUIRE(clipboard.text() == sample);

    clipboard.setText(original);
    REQUIRE(clipboard.text() == original);
}

TEST_CASE("an empty string clears the text", "[os][clipboard][state]") {
    requireStateChangesAllowed();
    Win32Clipboard clipboard;
    const std::string original = textOrSkip(clipboard);

    clipboard.setText("");
    REQUIRE(clipboard.text().empty());

    clipboard.setText(original);
}

// Design notes:
// - Only ClipboardBusyError skips: it is the one failure caused by another
//   process, after the product's own retries. A generic OsError (or any
//   other exception) is a real bug and still fails the test.
// - A busy clipboard is simulated through ClipboardOpenPolicy rather than
//   by holding it open: OpenClipboard(nullptr) on a second thread of the
//   same process does not lock out the first, so a real holder would need
//   a second process, and the injected opener is deterministic and runs
//   without a desktop (CI included).
