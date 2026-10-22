"""One-time account connection from a terminal (``scripts/connect_accounts.py``).

The windowless desktop app never asks for consent (see
:mod:`jarvis.integrations.consent`); this package is where it happens
instead, in the foreground, with the browser and the device code in front
of the user. Each service module signs in through the same token sources
the providers use, stores the result in the same :class:`TokenStore`, and
then proves it worked with one cheap API call per product.

Modules:
    google: client-JSON validation, installed-app consent, Gmail/Calendar check.
    microsoft: device-code sign-in rendered for a terminal, Outlook/Teams check.
    discord: bot-token, server membership, intent and permission diagnosis.
    status: what is configured and connected, without secrets.
    cli: the Typer commands behind ``connect_accounts.py`` and ``jarvis connect``.
"""
