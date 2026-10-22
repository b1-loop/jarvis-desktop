"""Connect your accounts once, in a terminal; the desktop app then signs in silently.

Run from the checkout:

    uv run python scripts/connect_accounts.py google
    uv run python scripts/connect_accounts.py microsoft [--teams]
    uv run python scripts/connect_accounts.py discord
    uv run python scripts/connect_accounts.py status
    uv run python scripts/connect_accounts.py disconnect google

The same commands are available as ``uv run jarvis connect ...``. Setup
steps for each service are in the README section "Connect your accounts".
The logic lives in :mod:`jarvis.integrations.connect.cli`.
"""

from __future__ import annotations

from jarvis.integrations.connect.cli import main

if __name__ == "__main__":
    main()
