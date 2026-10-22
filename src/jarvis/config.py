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

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from jarvis.inference.models import PIPER_VOICE_EN_GB, QWEN2_5_0_5B_INSTRUCT, WHISPER_TINY
from jarvis.integrations.documents import default_document_roots
from jarvis.integrations.web import DEFAULT_USER_AGENT
from jarvis.utils.errors import ConfigError

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

LLMProvider = Literal["anthropic", "ollama", "local"]
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


class RuntimeMode(StrEnum):
    """Which set of providers the pipeline is assembled from.

    CLOUD trades privacy and cost for quality and simplicity; LOCAL runs
    entirely offline. The trade-offs are documented in the CLOUD vs LOCAL ADR
    listed in docs/adr/README.md.
    """

    CLOUD = "cloud"
    LOCAL = "local"


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
    llm_provider: LLMProvider = "anthropic"
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


# Design notes:
# - Flat settings, not nested groups: the env file is flat, and a reviewer can
#   diff .env.example against this class line by line.
# - SecretStr for every credential so an accidental repr() or log never leaks
#   it; the redaction processor in utils/logging.py is the second line of
#   defence.
# - Model names default to the current Claude Sonnet because it has the best
#   latency/quality balance for a conversational assistant; Opus is a config
#   change away.
# - llm_provider is separate from RuntimeMode on purpose: Ollama is a third
#   option that runs locally but is not the fully offline LOCAL mode (it
#   still pairs with faster-whisper and edge-tts). The default Ollama model
#   is a small tool-capable one so tool calling works out of the box.
# - Local mode: thread counts of 0 let whisper.cpp / llama.cpp pick for the
#   machine. The LLM temperature defaults below the wrapper's 0.7 because a
#   0.5B model follows the JSON tool-call protocol (ADR-0013) noticeably
#   more reliably when it samples less freely.

# - Vision: 1920 px on the long edge is the 1080p class the Claude docs name
#   as the cost/quality balance for screenshots (the high-resolution tier
#   caps at 2576 px and downscales anything larger). The byte budget is
#   measured on the base64 text, which is how providers count, and 5 MB is
#   the strictest per-image limit across the Claude platforms. ADR-0016.
