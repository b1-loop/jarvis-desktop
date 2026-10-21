"""Who the user is: name, preferences, projects and habits.

The profile is not its own table. It is a view over the ``facts`` table
restricted to keys under ``profile.``, so the same storage, confidence and
source tracking apply whether a fact came from "my name is Bozhidar" or
from the agent noticing the user always asks for metric units. Rendering
the profile into the system prompt is what makes Jarvis feel like it
remembers the person between sessions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final, Protocol

from jarvis.memory.models import Fact

PROFILE_PREFIX: Final = "profile."
_NAME_KEY: Final = PROFILE_PREFIX + "name"
_PREFERENCE_PREFIX: Final = PROFILE_PREFIX + "preference."
_PROJECT_PREFIX: Final = PROFILE_PREFIX + "project."
_HABIT_PREFIX: Final = PROFILE_PREFIX + "habit."


class FactStore(Protocol):
    """The slice of the relational store the profile needs."""

    async def list_facts(self, prefix: str = "") -> list[Fact]:
        """Facts whose key starts with ``prefix``."""

    async def upsert_fact(
        self, key: str, value: str, *, source: str = "user", confidence: float = 1.0
    ) -> Fact:
        """Insert or update a fact."""

    async def delete_fact(self, key: str) -> bool:
        """Remove a fact; returns whether it existed."""


@dataclass(frozen=True, slots=True)
class UserProfile:
    """A snapshot of what Jarvis knows about the user.

    Attributes:
        name: How to address the user, if known.
        preferences: Named preferences, e.g. ``{"units": "metric"}``.
        projects: Project name to a one-line description.
        habits: Habit name to a one-line description, e.g.
            ``{"standup": "every weekday at 09:30"}``.
    """

    name: str | None = None
    preferences: Mapping[str, str] = field(default_factory=dict)
    projects: Mapping[str, str] = field(default_factory=dict)
    habits: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_facts(cls, facts: Iterable[Fact]) -> UserProfile:
        """Build a profile from ``profile.*`` facts; other keys are ignored."""
        name: str | None = None
        groups: dict[str, dict[str, str]] = {
            _PREFERENCE_PREFIX: {},
            _PROJECT_PREFIX: {},
            _HABIT_PREFIX: {},
        }
        for fact in facts:
            if fact.key == _NAME_KEY:
                name = fact.value
                continue
            for prefix, group in groups.items():
                if fact.key.startswith(prefix):
                    group[fact.key.removeprefix(prefix)] = fact.value
        return cls(
            name=name,
            preferences=groups[_PREFERENCE_PREFIX],
            projects=groups[_PROJECT_PREFIX],
            habits=groups[_HABIT_PREFIX],
        )

    @classmethod
    async def load(cls, store: FactStore) -> UserProfile:
        """Read the current profile from ``store``."""
        return cls.from_facts(await store.list_facts(PROFILE_PREFIX))

    async def update(
        self,
        store: FactStore,
        *,
        name: str | None = None,
        preferences: Mapping[str, str] | None = None,
        projects: Mapping[str, str] | None = None,
        habits: Mapping[str, str] | None = None,
        source: str = "user",
        confidence: float = 1.0,
    ) -> UserProfile:
        """Persist the given changes and return the resulting profile.

        Only the fields passed are written; existing entries under a
        mapping are kept unless overwritten, so "I also work on X" does not
        forget Y.
        """
        if name is not None:
            await store.upsert_fact(_NAME_KEY, name, source=source, confidence=confidence)
        for prefix, mapping in (
            (_PREFERENCE_PREFIX, preferences),
            (_PROJECT_PREFIX, projects),
            (_HABIT_PREFIX, habits),
        ):
            for key, value in (mapping or {}).items():
                await store.upsert_fact(prefix + key, value, source=source, confidence=confidence)
        return await self.load(store)

    async def forget(
        self,
        store: FactStore,
        *,
        name: bool = False,
        preferences: Iterable[str] = (),
        projects: Iterable[str] = (),
        habits: Iterable[str] = (),
    ) -> UserProfile:
        """Delete the named entries and return the resulting profile."""
        if name:
            await store.delete_fact(_NAME_KEY)
        for prefix, keys in (
            (_PREFERENCE_PREFIX, preferences),
            (_PROJECT_PREFIX, projects),
            (_HABIT_PREFIX, habits),
        ):
            for key in keys:
                await store.delete_fact(prefix + key)
        return await self.load(store)

    @property
    def is_empty(self) -> bool:
        """True when nothing at all is known about the user."""
        return self.name is None and not (self.preferences or self.projects or self.habits)

    def render(self) -> str:
        """Compact prompt text; empty string when nothing is known."""
        if self.is_empty:
            return ""
        lines = ["User profile:"]
        if self.name is not None:
            lines.append(f"- Name: {self.name}")
        for label, mapping in (
            ("Preferences", self.preferences),
            ("Projects", self.projects),
            ("Habits", self.habits),
        ):
            if mapping:
                items = "; ".join(f"{key}: {value}" for key, value in sorted(mapping.items()))
                lines.append(f"- {label}: {items}")
        return "\n".join(lines)


# Design notes:
# - The profile is immutable and `update`/`forget` return a fresh snapshot
#   read back from the store, so the object in hand always reflects what
#   is persisted and there is no cache to invalidate.
# - Key layout ``profile.<group>.<name>`` keeps the profile queryable with
#   one prefix scan and leaves ``facts`` free for everything else the agent
#   learns (``project.jarvis.repo``, ``weather.default_city`` ...).
# - render() sorts mapping entries so the prompt text is stable between
#   turns, which matters for prompt caching later.
