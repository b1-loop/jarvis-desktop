"""Command-line entry point: ``python -m jarvis`` or ``uv run jarvis``.

Parses the launch flags, loads settings, configures logging and hands
over to the composition root in :mod:`jarvis.core.bootstrap`, which builds
and runs the assistant. This module stays thin on purpose: every
construction decision lives in a small, tested builder there.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Annotated, Final

import typer

from jarvis import __version__
from jarvis.app.background import run_windowless
from jarvis.app.windowless import is_windowless
from jarvis.config import RuntimeMode, Settings
from jarvis.core.bootstrap import run_text_mode
from jarvis.core.voice_bootstrap import run_voice_mode
from jarvis.utils.errors import JarvisError
from jarvis.utils.logging import configure_logging, get_logger

EXIT_STARTUP_FAILED: Final = 2
"""Exit code when the assistant cannot be assembled (missing key, broken database)."""

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
    background: Annotated[
        bool,
        typer.Option(
            "--background",
            help="Run as a desktop app: tray icon only, log file, one instance (shortcuts).",
        ),
    ] = False,
    version: Annotated[  # noqa: ARG001 - consumed by the eager callback
        bool, typer.Option("--version", callback=_show_version, is_eager=True)
    ] = False,
) -> None:
    """Start the assistant."""
    settings = load_settings(mode)
    if is_windowless(background=background, stdout=sys.stdout):
        # No console to talk to: voice mode with the tray, logging to a file.
        raise typer.Exit(code=run_windowless(settings, wake_word=not no_wakeword))
    configure_logging(settings.log_level)
    options = LaunchOptions(
        mode=settings.mode, text_only=text_only, wake_word_enabled=not no_wakeword
    )
    _announce(options)
    try:
        asyncio.run(session(settings, options))
    except JarvisError as error:
        typer.echo(f"Jarvis could not start: {error}", err=True)
        raise typer.Exit(code=EXIT_STARTUP_FAILED) from error
    except KeyboardInterrupt:
        typer.echo("\nInterrupted. Goodbye.")


def _announce(options: LaunchOptions) -> None:
    get_logger(__name__).info(
        "jarvis.start",
        mode=options.mode.value,
        text_only=options.text_only,
        wake_word=options.wake_word_enabled,
    )
    typer.echo(
        f"Starting Jarvis (mode={options.mode.value}, text_only={options.text_only}, "
        f"wake_word={options.wake_word_enabled})."
    )


def session(settings: Settings, options: LaunchOptions) -> Coroutine[object, object, None]:
    """The conversation to run: voice by default, the terminal with ``--text-only``."""
    if options.text_only:
        return run_text_mode(settings)
    return run_voice_mode(settings, wake_word=options.wake_word_enabled)


def main() -> None:
    """Console-script entry point declared in ``pyproject.toml``."""
    app()


if __name__ == "__main__":
    main()


# Design notes:
# - Typer over argparse: typed options, enum validation and --help for free,
#   and the same declaration serves ``python -m jarvis`` and ``uv run jarvis``.
# - LaunchOptions is separate from Settings on purpose: command-line flags
#   describe this launch, settings describe the environment.
# - Voice is the default. When it cannot run (no native build, no models,
#   no microphone) run_voice_mode says why and continues in text mode
#   rather than pretending to listen.
# - Without a console (--background from the shortcuts, or pythonw where
#   sys.stdout is None) none of the terminal paths apply: jarvis.app runs
#   the windowless session, which logs to data/logs/jarvis.log and shows
#   problems in a notification and the tray instead. --text-only is
#   meaningless there and ignored.
# - Ctrl+C cancels the main task; the bootstrap's exit stack still closes
#   the database and connections before KeyboardInterrupt reaches here.
