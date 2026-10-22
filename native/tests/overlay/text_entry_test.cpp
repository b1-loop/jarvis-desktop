// The text field over a headless Dear ImGui context: characters arrive
// exactly as the Win32 backend delivers WM_CHAR (io.AddInputCharacterUTF16,
// one UTF-16 code unit per message) and must come out as UTF-8.

#include "text_entry.hpp"

#include <catch2/catch_test_macros.hpp>
#include <imgui.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

using jarvis::overlay::hud::TextEntry;
using jarvis::overlay::hud::trimmed;

namespace {

constexpr float kWidth = 400.0F;

/// One ImGui context with no renderer: fonts are built on demand.
class HeadlessImGui {
public:
    HeadlessImGui() : context_(ImGui::CreateContext()) {
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = ImVec2{800.0F, 600.0F};
        io.DeltaTime = 1.0F / 60.0F;
        io.IniFilename = nullptr;
        io.LogFilename = nullptr;
        // NOLINTNEXTLINE(bugprone-signed-bitwise,hicpp-signed-bitwise) - ImGui flags are ints
        io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    }
    ~HeadlessImGui() { ImGui::DestroyContext(context_); }
    HeadlessImGui(const HeadlessImGui&) = delete;
    HeadlessImGui& operator=(const HeadlessImGui&) = delete;
    HeadlessImGui(HeadlessImGui&&) = delete;
    HeadlessImGui& operator=(HeadlessImGui&&) = delete;

    /// Run one frame with the field drawn inside a window.
    TextEntry::Result frame(TextEntry& entry, bool focus) {
        ImGui::SetCurrentContext(context_);
        ++frames_;
        ImGui::NewFrame();
        ImGui::Begin("test");
        const TextEntry::Result result = entry.draw("##field", "hint", kWidth, focus);
        ImGui::End();
        ImGui::Render();
        return result;
    }

    /// Focus the field and let ImGui activate it.
    void focus(TextEntry& entry) {
        frame(entry, true);
        frame(entry, false);
    }

    /// Press and release @p key over two frames; the result of the press.
    TextEntry::Result press(TextEntry& entry, ImGuiKey key) {
        ImGui::GetIO().AddKeyEvent(key, true);
        const TextEntry::Result pressed = frame(entry, false);
        ImGui::GetIO().AddKeyEvent(key, false);
        frame(entry, false);
        return pressed;
    }

    /// Deliver @p units as WM_CHAR would, then let a frame consume them.
    void type(TextEntry& entry, std::u16string_view units) {
        for (const char16_t unit : units) {
            ImGui::GetIO().AddInputCharacterUTF16(static_cast<ImWchar16>(unit));
        }
        frame(entry, false);
    }

    [[nodiscard]] int frames() const noexcept { return frames_; }

private:
    ImGuiContext* context_;  // owned: destroyed in the destructor
    int frames_{0};
};

}  // namespace

TEST_CASE("Swedish letters typed through WM_CHAR come out as UTF-8", "[overlay][text]") {
    HeadlessImGui imgui;
    TextEntry entry;
    imgui.focus(entry);
    imgui.type(entry, u"Hej å ä ö ÅÄÖ");

    REQUIRE(entry.text() == "Hej \xC3\xA5 \xC3\xA4 \xC3\xB6 \xC3\x85\xC3\x84\xC3\x96");
    const TextEntry::Result submitted = imgui.press(entry, ImGuiKey_Enter);
    REQUIRE(submitted.submitted == std::optional<std::string>{"Hej \xC3\xA5 \xC3\xA4 \xC3\xB6 "
                                                              "\xC3\x85\xC3\x84\xC3\x96"});
    REQUIRE(entry.text().empty());
}

TEST_CASE("a surrogate pair from two WM_CHAR messages is one character", "[overlay][text]") {
    HeadlessImGui imgui;
    TextEntry entry;
    imgui.focus(entry);
    // U+1F600 arrives as the high and the low surrogate in two messages.
    imgui.type(entry, u"ok \xD83D\xDE00");
    REQUIRE(entry.text() == "ok \xF0\x9F\x98\x80");
}

TEST_CASE("Enter on blank text submits nothing; Escape cancels", "[overlay][text]") {
    HeadlessImGui imgui;
    TextEntry entry;
    imgui.focus(entry);
    imgui.type(entry, u"   ");
    REQUIRE_FALSE(imgui.press(entry, ImGuiKey_Enter).submitted.has_value());

    imgui.focus(entry);
    imgui.type(entry, u"draft");
    REQUIRE(imgui.press(entry, ImGuiKey_Escape).cancelled);
}

TEST_CASE("the field reports being edited while it has focus", "[overlay][text]") {
    HeadlessImGui imgui;
    TextEntry entry;
    REQUIRE_FALSE(imgui.frame(entry, false).active);
    imgui.frame(entry, true);
    REQUIRE(imgui.frame(entry, false).active);
    REQUIRE(imgui.frames() == 3);
}

TEST_CASE("set() never cuts a multi-byte character in half", "[overlay][text]") {
    TextEntry entry;
    std::string longText(TextEntry::kCapacity - 2, 'a');
    longText += "\xC3\xA5\xC3\xA5";  // the first a-ring straddles the capacity
    entry.set(longText);
    REQUIRE(entry.text().size() == TextEntry::kCapacity - 2);
    entry.set("\xC3\xA4");
    REQUIRE(entry.text() == "\xC3\xA4");
    entry.clear();
    REQUIRE(entry.text().empty());
}

TEST_CASE("trimmed strips surrounding white space only", "[overlay][text]") {
    REQUIRE(trimmed("  hej d\xC3\xA5\t\r\n") == "hej d\xC3\xA5");
    REQUIRE(trimmed(" \t ").empty());
    REQUIRE(trimmed("a b") == "a b");
}
