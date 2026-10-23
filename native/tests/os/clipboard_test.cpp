#include "jarvis/os/clipboard.hpp"

#include "jarvis/os/errors.hpp"

#include <catch2/catch_test_macros.hpp>

#include "os/desktop.hpp"
#include <windows.h>

#include <chrono>
#include <semaphore>
#include <string>
#include <thread>

using jarvis::os::ClipboardBusyError;
using jarvis::os::Win32Clipboard;
using jarvis::os::testing::requireDesktop;
using jarvis::os::testing::requireStateChangesAllowed;

namespace {

/// Keeps the clipboard open on a second thread, the way a clipboard manager
/// or a remote-desktop client does while it copies.
class ClipboardHolder {
public:
    ClipboardHolder()
        : thread_([this] {
              opened_ = OpenClipboard(nullptr) != FALSE;
              ready_.release();
              release_.acquire();
              if (opened_) {
                  CloseClipboard();
              }
          }) {
        ready_.acquire();
    }
    ~ClipboardHolder() { release(); }

    ClipboardHolder(const ClipboardHolder&) = delete;
    ClipboardHolder& operator=(const ClipboardHolder&) = delete;
    ClipboardHolder(ClipboardHolder&&) = delete;
    ClipboardHolder& operator=(ClipboardHolder&&) = delete;

    /// True when this holder got the clipboard (nobody else had it open).
    [[nodiscard]] bool opened() const noexcept { return opened_; }

    /// Close the clipboard and join; call from one thread only.
    void release() {
        if (thread_.joinable()) {
            release_.release();
            thread_.join();
        }
    }

private:
    std::binary_semaphore ready_{0};
    std::binary_semaphore release_{0};
    bool opened_ = false;
    std::jthread thread_;
};

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

void holdOrSkip(const ClipboardHolder& holder) {
    if (!holder.opened()) {
        SKIP("another application holds the clipboard, so this test cannot");
    }
}

}  // namespace

TEST_CASE("reading the clipboard never throws on a desktop", "[os][clipboard]") {
    requireDesktop();
    const Win32Clipboard clipboard;

    static_cast<void>(textOrSkip(clipboard));
}

TEST_CASE("a clipboard held by another thread is reported as busy", "[os][clipboard]") {
    requireDesktop();
    const ClipboardHolder holder;
    holdOrSkip(holder);
    const Win32Clipboard clipboard;

    REQUIRE_THROWS_AS(clipboard.text(), ClipboardBusyError);
}

TEST_CASE("a briefly held clipboard is read after retrying", "[os][clipboard]") {
    requireDesktop();
    ClipboardHolder holder;
    holdOrSkip(holder);
    constexpr auto kHold = std::chrono::milliseconds(50);
    const std::jthread releaser([&holder, kHold] {
        std::this_thread::sleep_for(kHold);
        holder.release();
    });
    const Win32Clipboard clipboard;

    REQUIRE_NOTHROW(clipboard.text());
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
// - The holder opens the clipboard on its own thread because OpenClipboard
//   denies access to every thread except the one that holds it, which is
//   exactly what another application looks like to Win32Clipboard.
// - In the "briefly held" test the releaser is declared after the holder,
//   so it is joined (and has released the clipboard) before the holder is
//   destroyed.
