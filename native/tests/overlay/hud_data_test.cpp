#include "jarvis/overlay/hud_data.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>

using jarvis::overlay::Appearance;
using jarvis::overlay::HistoryEntry;
using jarvis::overlay::HistoryRole;
using jarvis::overlay::HudDataLimits;
using jarvis::overlay::HudDataModel;
using jarvis::overlay::SettingItem;
using jarvis::overlay::SettingKind;
using jarvis::overlay::SettingsPage;
using jarvis::overlay::ThemeName;

TEST_CASE("history keeps the newest lines up to the limit", "[overlay][hud-data]") {
    HudDataModel model(HudDataLimits{.maxEntries = 3, .maxEntryChars = 100});
    for (int i = 0; i < 5; ++i) {
        model.addHistory(
            HistoryEntry{.role = HistoryRole::User, .text = std::to_string(i), .time = "12:00"});
    }
    const auto data = model.snapshot();
    REQUIRE(data.history->size() == 3);
    REQUIRE(data.history->front().text == "2");
    REQUIRE(data.history->back().text == "4");
    REQUIRE(data.historyRevision == 5);
}

TEST_CASE("long lines are cut at a character boundary", "[overlay][hud-data]") {
    HudDataModel model(HudDataLimits{.maxEntries = 10, .maxEntryChars = 4});
    model.addHistory(
        HistoryEntry{.role = HistoryRole::Jarvis, .text = "\xC3\xA5\xC3\xA4\xC3\xB6xyz"});
    const std::string& text = model.snapshot().history->front().text;
    REQUIRE(text.starts_with("\xC3\xA5\xC3\xA4\xC3\xB6"));
    REQUIRE(text.find("xyz") == std::string::npos);
}

TEST_CASE("a snapshot keeps its history while new lines arrive", "[overlay][hud-data]") {
    HudDataModel model;
    model.addHistory(HistoryEntry{.role = HistoryRole::User, .text = "first"});
    const auto before = model.snapshot();
    model.addHistory(HistoryEntry{.role = HistoryRole::Jarvis, .text = "second"});
    REQUIRE(before.history->size() == 1);
    REQUIRE(model.snapshot().history->size() == 2);
    model.clearHistory();
    REQUIRE(model.snapshot().history->empty());
    REQUIRE(before.history->size() == 1);
}

TEST_CASE("settings, mute, appearance and expand requests bump the revision",
          "[overlay][hud-data]") {
    HudDataModel model;
    const auto start = model.snapshot().revision;
    model.setMuted(true);
    model.setSettings(
        SettingsPage{.items = {SettingItem{
                         .key = "speak_replies", .kind = SettingKind::Toggle, .value = "true"}},
                     .restartPending = true,
                     .notice = "Saved."});
    const jarvis::overlay::Rgba orange{.r = 1.0F, .g = 0.5F, .b = 0.0F, .a = 1.0F};
    model.setAppearance(Appearance{.theme = ThemeName::Custom, .accent = orange});
    model.requestExpanded(true);

    const auto data = model.snapshot();
    REQUIRE(data.revision == start + 4);
    REQUIRE(data.muted);
    REQUIRE(data.settings->items.front().key == "speak_replies");
    REQUIRE(data.settings->restartPending);
    REQUIRE(data.settingsRevision == 1);
    REQUIRE(data.appearance.theme == ThemeName::Custom);
    REQUIRE(data.appearance.accent == orange);
    REQUIRE(data.expandRequest == true);
    REQUIRE(data.expandRequestId == 1);
}

TEST_CASE("a history limit of zero still keeps the newest line", "[overlay][hud-data]") {
    HudDataModel model(HudDataLimits{.maxEntries = 0});
    model.addHistory(HistoryEntry{.text = "a"});
    model.addHistory(HistoryEntry{.text = "b"});
    REQUIRE(model.snapshot().history->size() == 1);
    REQUIRE(model.snapshot().history->front().text == "b");
}

TEST_CASE("minimize requests and the minimize options travel as data", "[overlay][hud][orb]") {
    HudDataModel model;
    REQUIRE_FALSE(model.snapshot().minimizeRequest.has_value());
    REQUIRE(model.snapshot().hideInFullscreen);
    model.requestMinimized(true);
    model.requestMinimized(false);
    model.setAutoMinimize(-5.0F);
    model.setHideInFullscreen(false);

    const auto data = model.snapshot();
    REQUIRE(data.minimizeRequest == false);
    REQUIRE(data.minimizeRequestId == 2);
    REQUIRE(data.autoMinimizeSeconds == 0.0F);
    REQUIRE_FALSE(data.hideInFullscreen);
    model.setAutoMinimize(30.0F);
    REQUIRE(model.snapshot().autoMinimizeSeconds == 30.0F);
}
