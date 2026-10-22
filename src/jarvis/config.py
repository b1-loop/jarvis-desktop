"""Application settings, loaded once from the environment and ``.env``.

This module is the only place in Jarvis that reads environment variables.
Everything else receives a :class:`Settings` instance (or a slice of it)
through its constructor, which keeps modules testable and makes the full set
of knobs visible in one file. Field names mirror ``.env.example``.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from jarvis.inference.models import PIPER_VOICE_EN_GB, QWEN2_5_0_5B_INSTRUCT, WHISPER_TINY
from jarvis.integrations.documents import default_document_roots
from jarvis.integrations.web import DEFAULT_USER_AGENT
from jarvis.user_settings import UserSettingsSource
from jarvis.utils.errors import ConfigError

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

LLMProvider = Literal["anthropic", "ollama", "local"]
OverlayThemeName = Literal["cyan", "amber"]
OverlayIntensityName = Literal["calm", "normal", "max"]
OverlaySizeName = Literal["compact", "normal", "large"]
"""Which ``LLMClient`` answers: Claude, a local Ollama server, or in-process llama.cpp."""

DEFAULT_LOCAL_MODELS_DIR: Final = Path("models")
"""Where ``scripts/download_models.py --local`` puts the LOCAL-mode files by default."""

DEFAULT_SHELL_ALLOWLIST: Final[tuple[str, ...]] = (
    "git",
    "ls",
    "dir",
    "cat",
    "type",
    "python",
    "uv",
    "pip",
    "pytest",
    "ruff",
    "mypy",
    "npm",
    "npx",
    "node",
    "cargo",
    "where",
    "which",
)
"""Programs the shell tool runs without asking; anything else needs spoken confirmation."""

DEFAULT_CORE_TOOLS: Final[tuple[str, ...]] = ("current_time", "web_search")
"""Tools offered on every turn when the per-turn selection is active (ADR-0020)."""

SMALL_MODEL_TOOLS_PER_TURN: Final = 12
"""Tool schemas per request for Ollama and llama.cpp models; measured in ADR-0020."""

ALL_TOOLS_PER_TURN: Final = 0
"""``max_tools_per_turn`` value that offers every tool (the Claude default)."""


class RuntimeMode(StrEnum):
    """Which set of providers the pipeline is assembled from.

    CLOUD trades privacy and cost for quality and simplicity; LOCAL runs
    entirely offline. The trade-offs are documented in the CLOUD vs LOCAL ADR
    listed in docs/adr/README.md.
    """

    CLOUD = "cloud"
    LOCAL = "local"


class HudPosition(BaseModel):
    """A remembered HUD position (ADR-0024).

    Logical (96-DPI) pixels from the top-left of the named monitor's work
    area, so it survives DPI changes and monitors being reordered.

    Attributes:
        monitor: Device name (``MONITORINFOEX.szDevice``); an unknown one
            falls back to the primary monitor.
        x: Offset from the work area's left edge.
        y: Offset from the work area's top edge.
        width: Logical width (expanded panel only; 0 for the corner HUD).
        height: Logical height (expanded panel only; 0 for the corner HUD).
    """

    monitor: str
    x: float
    y: float
    width: float = Field(default=0.0, ge=0.0)
    height: float = Field(default=0.0, ge=0.0)


def default_documents_output_dir() -> Path:
    """Where ``draft_document`` writes when nothing is configured.

    A dedicated sub-folder of Documents keeps generated files together and
    apart from the user's own, so an overwrite can only hit a file in the
    folder Jarvis writes to.
    """
    return Path.home() / "Documents" / "Jarvis"


class Settings(BaseSettings):
    """All runtime configuration for Jarvis.

    Jarvis-specific knobs are read with the ``JARVIS_`` prefix; third-party
    credentials keep their conventional names (``ANTHROPIC_API_KEY`` ...) so
    they can be shared with other tools on the same machine.
    """

    model_config = SettingsConfigDict(
        env_prefix="JARVIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        # ``.env.example`` lists every key with an empty value; a copied
        # ``KEY=`` line must mean "not set", not "set to an empty string",
        # or providers would think they are configured.
        env_ignore_empty=True,
        extra="ignore",
        populate_by_name=True,
    )

    # Runtime
    mode: RuntimeMode = RuntimeMode.CLOUD
    wake_word: str = "hey jarvis"
    hotkey: str = "ctrl+alt+j"
    log_level: LogLevel = "INFO"
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")

    # LLM
    llm_provider: LLMProvider = "ollama"
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    llm_model: str = "claude-sonnet-5-5"
    vision_model: str = "claude-sonnet-5-5"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # Speech
    stt_model: str = "small"
    stt_device: str = "auto"
    stt_compute_type: str = "auto"
    tts_voice: str = "en-GB-RyanNeural"
    # Defaults are the pinned manifest files (jarvis.inference.models), so a
    # fresh `download_models.py --local` run is all LOCAL mode needs.
    local_llm_model: Path = DEFAULT_LOCAL_MODELS_DIR / QWEN2_5_0_5B_INSTRUCT.name
    local_stt_model: Path = DEFAULT_LOCAL_MODELS_DIR / WHISPER_TINY.name
    local_tts_model: Path = DEFAULT_LOCAL_MODELS_DIR / PIPER_VOICE_EN_GB.name

    # Integrations. ``None`` selects the Stub provider for that service.
    google_client_secrets_file: Path | None = Field(
        default=None, validation_alias="GOOGLE_CLIENT_SECRETS_FILE"
    )
    ms_client_id: str | None = Field(default=None, validation_alias="MS_CLIENT_ID")
    ms_tenant_id: str = Field(default="common", validation_alias="MS_TENANT_ID")
    spotify_client_id: str | None = Field(default=None, validation_alias="SPOTIFY_CLIENT_ID")
    spotify_client_secret: SecretStr | None = Field(
        default=None, validation_alias="SPOTIFY_CLIENT_SECRET"
    )
    slack_bot_token: SecretStr | None = Field(default=None, validation_alias="SLACK_BOT_TOKEN")
    discord_bot_token: SecretStr | None = Field(default=None, validation_alias="DISCORD_BOT_TOKEN")
    home_assistant_url: str | None = Field(default=None, validation_alias="HOME_ASSISTANT_URL")
    home_assistant_token: SecretStr | None = Field(
        default=None, validation_alias="HOME_ASSISTANT_TOKEN"
    )

    # Proactive
    routines_dir: Path = Path("routines")
    proactive_poll_seconds: int = Field(default=60, gt=0)

    # Agent
    confirm_destructive: bool = True
    shell_allowlist: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_SHELL_ALLOWLIST)
    )
    shell_timeout_seconds: int = 30
    shell_working_root: Path | None = None

    @field_validator("log_level", mode="before")
    @classmethod
    def _uppercase_log_level(cls, value: object) -> object:
        """Accept ``debug`` as well as ``DEBUG``; people type both."""
        return value.upper() if isinstance(value, str) else value

    @field_validator("shell_allowlist", mode="before")
    @classmethod
    def _split_allowlist(cls, value: object) -> object:
        """Accept ``git,python,uv`` from the env file instead of a JSON list."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("shell_working_root", mode="before")
    @classmethod
    def _empty_root_is_none(cls, value: object) -> object:
        """An empty ``JARVIS_SHELL_WORKING_ROOT=`` line means the home directory."""
        return None if value == "" else value

    def validate_for_mode(self) -> None:
        """Check that the credentials the selected mode needs are present.

        Kept out of the pydantic validators so that constructing ``Settings``
        never fails in tests or in ``--text-only`` LOCAL runs; the composition
        root calls this right before building providers.

        Raises:
            ConfigError: If CLOUD mode uses the Anthropic provider without a key.
        """
        needs_key = self.mode is RuntimeMode.CLOUD and self.llm_provider == "anthropic"
        if needs_key and self.anthropic_api_key is None:
            raise ConfigError(
                "ANTHROPIC_API_KEY is required in cloud mode with the anthropic provider. "
                "Set it in .env, set JARVIS_LLM_PROVIDER=ollama, or start with --mode local."
            )

    # Memory
    memory_recent_turns: int = 10
    memory_semantic_hits: int = 5

    # Integrations (extra), added with the providers in ADR-0009
    spotify_redirect_uri: str = Field(
        default="http://127.0.0.1:8888/callback", validation_alias="SPOTIFY_REDIRECT_URI"
    )
    discord_guild_id: str | None = Field(default=None, validation_alias="DISCORD_GUILD_ID")
    teams_enabled: bool = False
    """Add Teams chats to the Microsoft sign-in (``Chat.Read``, ``ChatMessage.Send``).

    Off by default: personal Microsoft accounts cannot use the Teams chat
    API, and a token request that includes scopes the account cannot grant
    can fail as a whole, taking Outlook down with it."""

    # Web and documents
    document_roots: Annotated[list[Path], NoDecode] = Field(default_factory=default_document_roots)
    web_user_agent: str = DEFAULT_USER_AGENT
    brave_search_api_key: SecretStr | None = Field(
        default=None, validation_alias="BRAVE_SEARCH_API_KEY"
    )

    @field_validator("document_roots", mode="before")
    @classmethod
    def _split_document_roots(cls, value: object) -> object:
        """Accept ``os.pathsep``-separated paths, the natural form for a .env line.

        An empty line (as in ``.env.example``) keeps the default roots.
        """
        if isinstance(value, str):
            roots = [Path(item.strip()) for item in value.split(os.pathsep) if item.strip()]
            return roots or default_document_roots()
        return value

    @field_validator("web_user_agent", mode="before")
    @classmethod
    def _default_empty_user_agent(cls, value: object) -> object:
        """An empty ``JARVIS_WEB_USER_AGENT=`` line keeps the default."""
        return DEFAULT_USER_AGENT if value == "" else value

    # Integration tools (step 8d)
    documents_output_dir: Path = Field(default_factory=default_documents_output_dir)

    @field_validator("documents_output_dir", mode="before")
    @classmethod
    def _default_empty_output_dir(cls, value: object) -> object:
        """An empty ``JARVIS_DOCUMENTS_OUTPUT_DIR=`` line keeps ``~/Documents/Jarvis``."""
        return default_documents_output_dir() if value == "" else value

    # Local mode (step 12b)
    stt_language: str | None = None
    local_stt_threads: int = Field(default=0, ge=0)
    local_llm_context_size: int = Field(default=4096, gt=0)
    local_llm_threads: int = Field(default=0, ge=0)
    local_llm_max_tokens: int = Field(default=512, gt=0)
    local_llm_temperature: float = Field(default=0.4, ge=0.0)

    @field_validator("stt_language", mode="before")
    @classmethod
    def _empty_language_is_auto(cls, value: object) -> object:
        """An empty ``JARVIS_STT_LANGUAGE=`` line means "detect the language"."""
        return None if value == "" else value

    # Vision (step 11)
    vision_enabled: bool = True
    vision_max_long_edge: int = Field(default=1920, ge=256, le=8000)
    vision_max_image_bytes: int = Field(default=5_000_000, ge=100_000)
    vision_save_screenshots: bool = True
    ollama_vision_model: str | None = None

    @field_validator("ollama_vision_model", mode="before")
    @classmethod
    def _empty_vision_model_is_none(cls, value: object) -> object:
        """An empty ``JARVIS_OLLAMA_VISION_MODEL=`` line means Ollama has no vision model."""
        return None if value == "" else value

    # Agent loop (step 6b)
    assistant_name: str = Field(default="Jarvis", min_length=1)
    persona_tone: str = "calm, concise and dryly witty, like a capable butler"
    max_tool_rounds: int = Field(default=6, ge=1)

    # Voice (step 7)
    barge_in_echo_guard_ms: int = Field(default=400, ge=0)
    vad_min_speech_ms: int = Field(default=250, ge=32)
    wake_word_threshold: float = Field(default=0.8, gt=0.0, lt=1.0)
    listen_timeout_seconds: float = Field(default=6.0, gt=0)
    max_utterance_seconds: float = Field(default=20.0, gt=0)
    min_transcript_chars: int = Field(default=2, ge=1)
    overlay_enabled: bool = True
    tray_enabled: bool = True
    dashboard_enabled: bool = True
    proactive_enabled: bool = True

    # Agent hardening
    max_tools_per_turn: int | None = Field(default=None, ge=0)
    core_tools: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: list(DEFAULT_CORE_TOOLS)
    )

    @field_validator("core_tools", mode="before")
    @classmethod
    def _split_core_tools(cls, value: object) -> object:
        """Accept ``current_time,web_search`` from the env file."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    # Desktop app
    dashboard_start_visible: bool = True
    log_file_max_bytes: int = Field(default=1_000_000, ge=10_000)
    log_file_backups: int = Field(default=3, ge=0)
    ollama_autostart: bool = True
    ollama_start_timeout_seconds: float = Field(default=20.0, gt=0)

    def tools_per_turn(self) -> int:
        """How many tool schemas one request may carry; 0 means all of them.

        ``JARVIS_MAX_TOOLS_PER_TURN`` wins when set. Otherwise Claude gets
        every tool and the small models (Ollama, llama.cpp, and LOCAL mode,
        which never uses Claude) get a short, relevant list.
        """
        if self.max_tools_per_turn is not None:
            return self.max_tools_per_turn
        uses_claude = self.mode is RuntimeMode.CLOUD and self.llm_provider == "anthropic"
        return ALL_TOOLS_PER_TURN if uses_claude else SMALL_MODEL_TOOLS_PER_TURN

    # Overlay animations
    overlay_boot_animation: bool = True
    overlay_theme: OverlayThemeName = "cyan"

    # Overlay richness
    overlay_intensity: OverlayIntensityName = "normal"
    overlay_size: OverlaySizeName = "large"

    # Integrations demo data
    demo_integrations: bool = False
    """Serve the ``Stub*`` sample mail, events and chats when no account is
    connected. Off by default: unconnected services say so instead."""

    # Interactive HUD
    overlay_interactive: bool = True
    """Clickable HUD: hover controls, typing, dragging, expanded view (ADR-0024)."""
    speak_replies: bool = True
    """Speak answers aloud; off keeps a typed conversation silent (text only)."""
    wake_word_enabled: bool = True
    """Listen for the wake word; ``--no-wakeword`` also turns it off for one launch."""
    overlay_position: HudPosition | None = None
    """Where the user dragged the corner HUD; None anchors it to its corner."""
    overlay_expanded_bounds: HudPosition | None = None
    """Position and size of the expanded panel; None opens it at 900 x 640."""
    user_settings_enabled: bool = True
    """Layer ``data_dir/user_settings.json`` (the HUD's choices) over ``.env``."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Insert the HUD's ``user_settings.json`` between the environment and ``.env``.

        Precedence, highest first: init kwargs > environment variables >
        user_settings.json > .env > defaults (see ``jarvis.user_settings``
        and ADR-0024 for why).
        """
        user = UserSettingsSource(settings_cls, (init_settings, env_settings, dotenv_settings))
        return init_settings, env_settings, user, dotenv_settings, file_secret_settings


# Design notes:
# - Flat settings, not nested groups: the env file is flat, and a reviewer can
#   diff .env.example against this class line by line.
# - SecretStr for every credential so an accidental repr() or log never leaks
#   it; the redaction processor in utils/logging.py is the second line of
#   defence.
# - The LLM provider defaults to Ollama with qwen2.5:7b: free, local and
#   good enough for tool calling once the tool list is trimmed per turn
#   (ADR-0020). Claude is the opt-in upgrade (JARVIS_LLM_PROVIDER=anthropic,
#   ADR-0013); its model names default to the current Sonnet because it has
#   the best latency/quality balance for a conversational assistant.
# - llm_provider is separate from RuntimeMode on purpose: Ollama is a third
#   option that runs locally but is not the fully offline LOCAL mode (it
#   still pairs with faster-whisper and edge-tts). The default Ollama model
#   is a small tool-capable one so tool calling works out of the box.
# - The echo guard defaults to 400 ms: long enough to cover the burst of
#   Jarvis's own first syllable reaching the microphone through speakers,
#   short enough that a user who interrupts at once is heard. There is no
#   echo cancellation (ADR-0019), so headphones remain the real fix.
# - Local mode: thread counts of 0 let whisper.cpp / llama.cpp pick for the
#   machine. The LLM temperature defaults below the wrapper's 0.7 because a
#   0.5B model follows the JSON tool-call protocol (ADR-0013) noticeably
#   more reliably when it samples less freely.

# - Vision: 1920 px on the long edge is the 1080p class the Claude docs name
#   as the cost/quality balance for screenshots (the high-resolution tier
#   caps at 2576 px and downscales anything larger). The byte budget is
#   measured on the base64 text, which is how providers count, and 5 MB is
#   the strictest per-image limit across the Claude platforms. ADR-0016.
