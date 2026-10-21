"""Application settings, loaded once from the environment and ``.env``.

This module is the only place in Jarvis that reads environment variables.
Everything else receives a :class:`Settings` instance (or a slice of it)
through its constructor, which keeps modules testable and makes the full set
of knobs visible in one file. Field names mirror ``.env.example``.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from jarvis.utils.errors import ConfigError

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

LLMProvider = Literal["anthropic", "ollama", "local"]
"""Which ``LLMClient`` answers: Claude, a local Ollama server, or llama.cpp (step 12)."""


class RuntimeMode(StrEnum):
    """Which set of providers the pipeline is assembled from.

    CLOUD trades privacy and cost for quality and simplicity; LOCAL runs
    entirely offline. The trade-offs are documented in ADR-0009.
    """

    CLOUD = "cloud"
    LOCAL = "local"


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
    local_llm_model: Path = Path("models/llama.gguf")
    local_stt_model: Path = Path("models/ggml-base.bin")
    local_tts_model: Path = Path("models/piper-voice.onnx")

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

    @field_validator("log_level", mode="before")
    @classmethod
    def _uppercase_log_level(cls, value: object) -> object:
        """Accept ``debug`` as well as ``DEBUG``; people type both."""
        return value.upper() if isinstance(value, str) else value

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
