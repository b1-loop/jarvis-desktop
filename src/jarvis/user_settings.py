"""The user's own settings: ``data_dir/user_settings.json``, written by the HUD.

``.env`` is the installer's configuration; choices the user makes later in
the HUD's settings page (a voice, the theme, where the HUD sits) are kept
apart in a small JSON file under ``Settings.data_dir`` and layered over
``.env`` at startup by :class:`UserSettingsSource`, a pydantic-settings
source. The documented precedence (ADR-0024), highest first:

1. keyword arguments to ``Settings(...)`` (tests, ``--mode``),
2. environment variables (``JARVIS_*`` set for this launch),
3. ``user_settings.json`` (the HUD),
4. ``.env``,
5. the field defaults.

:class:`UserSettingsStore` reads and writes the file atomically; both the
source and the HUD validate each value against its field, so one bad entry
is logged and skipped instead of stopping Jarvis from starting.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final

from pydantic import SecretStr, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from pydantic_settings import PydanticBaseSettingsSource

from jarvis.utils.errors import ConfigError
from jarvis.utils.logging import get_logger

if TYPE_CHECKING:
    from pydantic_settings import BaseSettings

log = get_logger(__name__)

USER_SETTINGS_FILENAME: Final = "user_settings.json"
DATA_DIR_FIELD: Final = "data_dir"
ENABLED_FIELD: Final = "user_settings_enabled"
"""Setting that turns the file off (the test suite sets it to false)."""
_LOCATING_FIELDS: Final = frozenset({DATA_DIR_FIELD, ENABLED_FIELD})
"""Decide where (and whether) the file is read, so the file cannot set them."""
_TEMP_SUFFIX: Final = ".tmp"
_FALSE_WORDS: Final = frozenset({"0", "false", "no", "off"})

SourceState = Callable[[], Mapping[str, Any]]
"""A pydantic-settings source called for its values (init kwargs, environment, .env)."""


class UserSettingsError(ConfigError):
    """``user_settings.json`` exists but is not a JSON object."""


def user_settings_path(data_dir: Path) -> Path:
    """Where the user settings live for ``data_dir``."""
    return data_dir / USER_SETTINGS_FILENAME


class UserSettingsStore:
    """Reads and writes one ``user_settings.json``.

    Args:
        path: The file; its directory is created on the first save.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """The file this store reads and writes."""
        return self._path

    def load(self) -> dict[str, Any]:
        """Everything in the file; empty when there is no file yet.

        Raises:
            UserSettingsError: If the file is not valid JSON or not an object.
        """
        if not self._path.is_file():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise UserSettingsError(f"{self._path} cannot be read: {error}") from error
        if not isinstance(data, dict):
            raise UserSettingsError(f"{self._path} must hold a JSON object")
        return data

    def save(self, values: Mapping[str, Any]) -> None:
        """Replace the file with ``values`` via a temporary file and a rename.

        A crash mid-write therefore leaves the previous file, never half a
        JSON document that would be ignored at the next start.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(self._path.name + _TEMP_SUFFIX)
        temporary.write_text(json.dumps(dict(values), indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self._path)

    def update(self, key: str, value: Any) -> dict[str, Any]:
        """Set one key and save; returns the whole new content."""
        values = self.load()
        values[key] = value
        self.save(values)
        return values


def editable_fields(settings_cls: type[BaseSettings]) -> dict[str, FieldInfo]:
    """The fields a user settings file may set: every field that is not a secret.

    Credentials stay in the environment or ``.env`` (CLAUDE.md: API keys
    only from environment variables), so a file the HUD writes can never
    hold one.
    """
    return {
        name: info
        for name, info in settings_cls.model_fields.items()
        if not _holds_secret(info) and name not in _LOCATING_FIELDS
    }


def validate_field_value(info: FieldInfo, value: Any) -> Any:
    """``value`` validated against one field's type and constraints.

    Raises:
        ValidationError: If the value does not fit the field.
    """
    # Annotated needs at least one metadata item; constraints (ge, le) live there.
    target: Any = Annotated[info.annotation, *info.metadata] if info.metadata else info.annotation
    adapter: TypeAdapter[Any] = TypeAdapter(target)
    return adapter.validate_python(value)


class UserSettingsSource(PydanticBaseSettingsSource):
    """pydantic-settings source over ``user_settings.json``.

    Args:
        settings_cls: The settings class being built.
        earlier: The sources that may say where ``data_dir`` is and whether
            the file is enabled, highest precedence first (init kwargs,
            environment, ``.env``); consulted lazily when the source runs.
    """

    def __init__(self, settings_cls: type[BaseSettings], earlier: tuple[SourceState, ...]) -> None:
        super().__init__(settings_cls)
        self._earlier = earlier

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        """Unused: :meth:`__call__` reads the whole file at once."""
        del field
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """The valid, non-secret values of the file, keyed by field name."""
        if not self._enabled():
            return {}
        store = UserSettingsStore(user_settings_path(self._data_dir()))
        try:
            raw = store.load()
        except UserSettingsError:
            log.exception("user_settings.unreadable", path=str(store.path))
            return {}
        return self._valid(raw, store.path)

    def _valid(self, raw: Mapping[str, Any], path: Path) -> dict[str, Any]:
        fields = editable_fields(self.settings_cls)
        values: dict[str, Any] = {}
        for key, value in raw.items():
            info = fields.get(key)
            if info is None:
                log.warning("user_settings.unknown_key", key=key, path=str(path))
                continue
            try:
                values[key] = validate_field_value(info, value)
            except ValidationError as error:
                log.warning("user_settings.invalid", key=key, error=str(error), path=str(path))
        return values

    def _first(self, name: str) -> Any:
        for source in self._earlier:
            value = source().get(name)
            if value is not None and value != "":
                return value
        return None

    def _enabled(self) -> bool:
        value = self._first(ENABLED_FIELD)
        if value is None:
            return True
        return str(value).strip().lower() not in _FALSE_WORDS

    def _data_dir(self) -> Path:
        value = self._first(DATA_DIR_FIELD)
        if value is None:
            default = self.settings_cls.model_fields[DATA_DIR_FIELD].default
            return Path(default)
        return Path(value)


def _holds_secret(info: FieldInfo) -> bool:
    annotation = info.annotation
    return annotation is SecretStr or SecretStr in getattr(annotation, "__args__", ())


# Design notes:
# - A separate file instead of rewriting .env: .env is hand-edited, may hold
#   comments and secrets, and is the installer's baseline; the HUD only ever
#   writes plain values it validated, into a file nobody needs to read.
# - Environment variables beat the file because they are explicit for one
#   launch (a test, a debugging session, CI); the file beats .env because
#   it holds the user's later, deliberate choices. ADR-0024 has the table.
# - data_dir itself is resolved from the higher sources and .env before the
#   file is read; a data_dir (or the on/off switch) inside the file would be
#   circular, so those keys are reported as unknown and ignored.
# - Each value is validated alone, so a hand-edited typo costs one setting
#   (logged) rather than the whole start-up, which would otherwise fail in
#   pydantic with no way to reach the HUD that could fix it.
