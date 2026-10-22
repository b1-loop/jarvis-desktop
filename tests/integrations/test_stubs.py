"""Interface contract tests, run against every stub."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from jarvis.integrations.base import (
    CalendarProvider,
    ChatProvider,
    EmailProvider,
    MusicProvider,
    Provider,
    ProviderError,
    SmartHomeProvider,
    TimeRange,
)
from jarvis.integrations.stubs import (
    StubCalendarProvider,
    StubChatProvider,
    StubEmailProvider,
    StubMusicProvider,
    StubSmartHomeProvider,
)

NOW = datetime(2026, 10, 21, 8, 30, tzinfo=UTC)

ALL_STUBS: list[tuple[type[Provider], type[Provider]]] = [
    (StubEmailProvider, EmailProvider),
    (StubCalendarProvider, CalendarProvider),
    (StubMusicProvider, MusicProvider),
    (StubChatProvider, ChatProvider),
    (StubSmartHomeProvider, SmartHomeProvider),
]


@pytest.mark.parametrize(("stub_cls", "interface"), ALL_STUBS)
async def test_every_stub_implements_its_interface_and_says_so(
    stub_cls: type[Provider], interface: type[Provider]
) -> None:
    stub = stub_cls()

    assert isinstance(stub, interface)
    assert "Stub" in stub_cls.__name__
    assert "stub" in stub.provider_name
    status = await stub.status()
    assert status.stub
    assert not status.configured
    assert not status.authenticated
    assert status.last_error is None


# -- email --------------------------------------------------------------------


async def test_email_stub_lists_unread_newest_first_and_records_calls() -> None:
    stub = StubEmailProvider()

    unread = await stub.list_unread(limit=5)

    assert [m.id for m in unread] == ["stub-mail-1", "stub-mail-2"]
    assert stub.calls == [("list_unread", {"limit": 5})]


async def test_email_stub_search_get_and_missing_id() -> None:
    stub = StubEmailProvider()

    assert [m.id for m in await stub.search("design review")] == ["stub-mail-1"]
    assert (await stub.get("stub-mail-2")).subject.startswith("[jarvis-desktop]")
    with pytest.raises(ProviderError, match="no message"):
        await stub.get("nope")


async def test_email_stub_drafts_then_sends_without_leaving_the_process() -> None:
    stub = StubEmailProvider()
    original = await stub.get("stub-mail-1")

    draft = await stub.draft_reply(original, "Thursday works for me.")
    sent_id = await stub.send(draft)

    assert draft.subject == "Re: Design review moved to Thursday"
    assert draft.to == (original.sender,)
    assert draft.in_reply_to == original.id
    assert sent_id == "stub-sent-1"
    assert stub.sent == [draft]
    assert draft.id not in stub.drafts


async def test_email_stub_does_not_double_the_re_prefix() -> None:
    stub = StubEmailProvider()
    message = (await stub.list_unread())[0]
    replied = await stub.draft_reply(message, "x")

    again = await stub.draft_reply(replace(message, subject=replied.subject), "y")

    assert again.subject == replied.subject


# -- calendar -----------------------------------------------------------------


async def test_calendar_stub_agenda_and_create_event() -> None:
    stub = StubCalendarProvider(now=NOW)
    day = NOW.replace(hour=0, minute=0)

    created = await stub.create_event(
        "Coffee", day + timedelta(hours=15), day + timedelta(hours=15, minutes=30), location="Cafe"
    )
    agenda = await stub.agenda(TimeRange(day + timedelta(hours=12), day + timedelta(hours=16)))

    assert created.id == "stub-event-4"
    assert [e.title for e in agenda] == ["1:1 with Maya", "Coffee"]


# -- music --------------------------------------------------------------------


async def test_music_stub_play_pause_resume_queue_next() -> None:
    stub = StubMusicProvider()

    played = await stub.play("blue in green")
    await stub.pause()
    paused = await stub.now_playing()
    await stub.resume()
    queued = await stub.queue("weightless")
    await stub.next_track()
    after_skip = await stub.now_playing()
    await stub.next_track()

    assert played.title == "Blue in Green"
    assert paused is not None and not paused.is_playing
    assert queued.artist == "Marconi Union"
    assert after_skip is not None and after_skip.track == queued
    assert await stub.now_playing() is None


async def test_music_stub_search_by_uri_and_unknown_query() -> None:
    stub = StubMusicProvider()

    assert [t.id for t in await stub.search("debussy")] == ["stub-track-1"]
    assert (await stub.play("stub:track:2")).id == "stub-track-2"
    with pytest.raises(ProviderError, match="nothing matches"):
        await stub.play("polka")


# -- chat ---------------------------------------------------------------------


async def test_chat_stub_filters_and_sends() -> None:
    stub = StubChatProvider()

    everything = await stub.list_recent()
    dev_only = await stub.list_recent("dev")
    mentions = await stub.list_recent(mentions_only=True)
    unread = await stub.list_recent(unread_only=True, limit=1)
    sent = await stub.send("C-general", "On my way")

    assert [m.id for m in everything] == ["stub-msg-1", "stub-msg-2", "stub-msg-3"]
    assert [m.id for m in dev_only] == ["stub-msg-2", "stub-msg-3"]
    assert [m.id for m in mentions] == ["stub-msg-2"]
    assert [m.id for m in unread] == ["stub-msg-1"]
    assert sent.channel_id == "C-general" and sent.text == "On my way"
    assert stub.messages[-1] == sent


# -- smart home ---------------------------------------------------------------


async def test_smart_home_stub_controls_devices() -> None:
    stub = StubSmartHomeProvider()

    lights = await stub.list_devices("light")
    bedroom_on = await stub.turn_on("light.bedroom")
    dimmed = await stub.set_brightness("light.bedroom", 30)
    off = await stub.turn_off("light.living_room")
    await stub.set_scene("scene.movie_night")

    assert [d.id for d in lights] == ["light.living_room", "light.bedroom"]
    assert bedroom_on.is_on and bedroom_on.brightness == 100
    assert dimmed.brightness == 30 and (await stub.state("light.bedroom")).brightness == 30
    assert not off.is_on and off.brightness == 0
    assert ("set_scene", {"scene_id": "scene.movie_night"}) in stub.calls


async def test_smart_home_stub_rejects_unknown_and_non_dimmable() -> None:
    stub = StubSmartHomeProvider()

    with pytest.raises(ProviderError, match="unknown device"):
        await stub.state("light.garage")
    with pytest.raises(ProviderError, match="not dimmable"):
        await stub.set_brightness("switch.desk_lamp", 50)
    assert (await stub.turn_off("switch.desk_lamp")).brightness is None
