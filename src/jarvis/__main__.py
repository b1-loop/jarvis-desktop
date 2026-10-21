"""Command-line entry point: ``python -m jarvis`` or ``uv run jarvis``.

This module is the composition root. It is the one place that is allowed to
know about concrete providers, because it is where they are chosen from the
configuration and injected into the orchestrator. Until the orchestrator
lands (step 6) it resolves the configuration, configures logging and exits
with an explicit message rather than pretending to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Final

import typer

from jarvis import __version__
from jarvis.config import RuntimeMode, Settings
from jarvis.utils.logging import configure_logging, get_logger

EXIT_NOT_IMPLEMENTED: Final = 2
"""Exit code used while the pipeline is still being built; never 0, never silent."""

app = typer.Typer(
    add_completion=False,
    rich_markup_mode=None,
    help="Jarvis: a voice-controlled personal AI assistant for the desktop.",
)


@dataclass(frozen=True, slots=True)
class LaunchOptions:
    """Resolved command-line choices handed to the orchestrator.

    Attributes:
        mode: Which provider set to assemble.
        text_only: Skip audio entirely and converse in the terminal.
        wake_word_enabled: Listen for the wake word; when False the global
            hotkey is the only way to start a turn.
    """

    mode: RuntimeMode
    text_only: bool
    wake_word_enabled: bool


def load_settings(mode_override: RuntimeMode | None) -> Settings:
    """Load settings from the environment, letting ``--mode`` win over ``.env``."""
    settings = Settings()
    if mode_override is None:
        return settings
    return settings.model_copy(update={"mode": mode_override})


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"jarvis {__version__}")
        raise typer.Exit


@app.command()
def run(
    mode: Annotated[
        RuntimeMode | None,
        typer.Option("--mode", case_sensitive=False, help="Provider set: cloud or local."),
    ] = None,
    text_only: Annotated[
        bool, typer.Option("--text-only", help="Type instead of talk; no audio devices needed.")
    ] = False,
    no_wakeword: Annotated[
        bool, typer.Option("--no-wakeword", help="Disable the wake word; use the hotkey only.")
    ] = False,
    version: Annotated[  # noqa: ARG001 - consumed by the eager callback
        bool, typer.Option("--version", callback=_show_version, is_eager=True)
    ] = False,
) -> None:
    """Start the assistant."""
    settings = load_settings(mode)
    configure_logging(settings.log_level)
    options = LaunchOptions(
        mode=settings.mode, text_only=text_only, wake_word_enabled=not no_wakeword
    )
    get_logger(__name__).info(
        "jarvis.start",
        mode=options.mode.value,
        text_only=options.text_only,
        wake_word=options.wake_word_enabled,
    )
    typer.echo(
        f"Configuration resolved (mode={options.mode.value}, text_only={options.text_only}, "
        f"wake_word={options.wake_word_enabled}). The pipeline arrives in step 6; "
        "nothing runs yet."
    )
    raise typer.Exit(code=EXIT_NOT_IMPLEMENTED)


def main() -> None:
    """Console-script entry point declared in ``pyproject.toml``."""
    app()


if __name__ == "__main__":
    main()


# Design notes:
# - Typer over argparse: typed options, enum validation and --help for free,
#   and the same declaration serves ``python -m jarvis`` and ``uv run jarvis``.
# - LaunchOptions is separate from Settings on purpose: command-line flags
#   describe this launch, settings describe the environment. The orchestrator
#   will take both.
