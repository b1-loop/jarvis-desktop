from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from jarvis.memory.models import Fact
from jarvis.memory.sqlite_store import SqliteStore
from jarvis.memory.user_profile import PROFILE_PREFIX, UserProfile
from tests.memory.fakes import EPOCH


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteStore]:
    async with SqliteStore(tmp_path / "memory.db", full_text_search=False) as opened:
        yield opened


def fact(key: str, value: str) -> Fact:
    return Fact(key, value, "user", 1.0, EPOCH, EPOCH)


def test_empty_profile_renders_nothing() -> None:
    profile = UserProfile()

    assert profile.is_empty
    assert profile.render() == ""


def test_from_facts_groups_keys_and_ignores_unrelated_ones() -> None:
    profile = UserProfile.from_facts(
        [
            fact("profile.name", "Bozhidar"),
            fact("profile.preference.units", "metric"),
            fact("profile.project.jarvis", "voice assistant"),
            fact("profile.habit.standup", "weekdays 09:30"),
            fact("weather.city", "Sofia"),
        ]
    )

    assert profile.name == "Bozhidar"
    assert profile.preferences == {"units": "metric"}
    assert profile.projects == {"jarvis": "voice assistant"}
    assert profile.habits == {"standup": "weekdays 09:30"}


def test_render_is_compact_and_sorted() -> None:
    profile = UserProfile(
        name="Bozhidar",
        preferences={"tone": "brief", "units": "metric"},
        projects={"jarvis": "voice assistant"},
    )

    assert profile.render() == (
        "User profile:\n"
        "- Name: Bozhidar\n"
        "- Preferences: tone: brief; units: metric\n"
        "- Projects: jarvis: voice assistant"
    )


async def test_profile_round_trips_through_the_fact_store(store: SqliteStore) -> None:
    profile = await UserProfile.load(store)
    assert profile.is_empty

    updated = await profile.update(
        store,
        name="Bozhidar",
        preferences={"units": "metric"},
        projects={"jarvis": "voice assistant"},
        habits={"standup": "09:30"},
    )
    reloaded = await UserProfile.load(store)

    assert updated == reloaded
    assert reloaded.name == "Bozhidar"
    assert reloaded.habits == {"standup": "09:30"}
    assert {f.key for f in await store.list_facts(PROFILE_PREFIX)} == {
        "profile.name",
        "profile.preference.units",
        "profile.project.jarvis",
        "profile.habit.standup",
    }


async def test_update_merges_instead_of_replacing(store: SqliteStore) -> None:
    profile = await UserProfile().update(store, projects={"jarvis": "assistant"})

    profile = await profile.update(store, projects={"thesis": "ml paper"}, source="llm")

    assert profile.projects == {"jarvis": "assistant", "thesis": "ml paper"}
    thesis = await store.get_fact("profile.project.thesis")
    assert thesis is not None
    assert thesis.source == "llm"


async def test_forget_removes_only_what_was_named(store: SqliteStore) -> None:
    profile = await UserProfile().update(
        store, name="B", preferences={"a": "1", "b": "2"}, habits={"gym": "mornings"}
    )

    profile = await profile.forget(store, name=True, preferences=["a"], habits=["gym"])

    assert profile.name is None
    assert profile.preferences == {"b": "2"}
    assert profile.habits == {}
