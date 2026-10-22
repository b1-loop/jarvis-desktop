"""Gmail provider over the Gmail REST API.

Authentication comes from :class:`~jarvis.integrations.google_auth.GoogleAuth`
(or any :class:`TokenSource`); the API itself is called with httpx rather
than the discovery-based ``google-api-python-client`` so requests are plain,
typed and testable with ``httpx.MockTransport``. Scopes are the two smallest
that cover the feature list: read, and compose (create drafts and send
them), never modify labels or delete. ``gmail.send`` alone is not enough:
``drafts.create`` and ``drafts.send`` both require ``gmail.compose``.
"""

from __future__ import annotations

import asyncio
import base64
from datetime import UTC, datetime
from email.message import EmailMessage as MimeMessage
from typing import Any

import httpx

from jarvis.integrations.base import EmailDraft, EmailMessage, EmailProvider, ProviderStatus
from jarvis.integrations.rest import RestSession
from jarvis.integrations.tokens import TokenSource

GMAIL_SCOPES = (
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
)
BASE_URL = "https://gmail.googleapis.com/gmail/v1/users/me"

_METADATA_HEADERS = ("From", "To", "Subject", "Message-ID")
_MS_PER_SECOND = 1000


class GmailProvider(EmailProvider):
    """Email over Gmail. See the module docstring for the design."""

    provider_name = "gmail"
    scopes = GMAIL_SCOPES

    def __init__(self, http: httpx.AsyncClient, token_source: TokenSource) -> None:
        self._rest = RestSession(
            http, provider=self.provider_name, base_url=BASE_URL, token_source=token_source
        )

    async def status(self) -> ProviderStatus:
        """Session-level status; see :meth:`RestSession.status`."""
        return self._rest.status()

    async def list_unread(self, limit: int = 10) -> list[EmailMessage]:
        """Unread inbox messages, newest first (Gmail's default order)."""
        return await self._query("is:unread in:inbox", limit)

    async def search(self, query: str, limit: int = 10) -> list[EmailMessage]:
        """Gmail search syntax, e.g. ``from:maya newer_than:7d``."""
        return await self._query(query, limit)

    async def get(self, message_id: str) -> EmailMessage:
        """Full message including the plain-text body."""
        return await self._fetch(message_id, "full")

    async def draft_reply(self, message: EmailMessage, body: str) -> EmailDraft:
        """Create a server-side draft in the original thread."""
        raw = _encode_mime(_reply_mime(message, body))
        data = await self._rest.request(
            "POST", "/drafts", json={"message": {"threadId": message.thread_id, "raw": raw}}
        )
        return EmailDraft(
            id=data["id"],
            to=(message.sender,),
            subject=_reply_subject(message.subject),
            body=body,
            in_reply_to=message.id,
            thread_id=message.thread_id,
        )

    async def send(self, draft: EmailDraft) -> str:
        """Send an existing draft; Gmail returns the sent message's id."""
        data = await self._rest.request("POST", "/drafts/send", json={"id": draft.id})
        return str(data["id"])

    async def _query(self, query: str, limit: int) -> list[EmailMessage]:
        listing = await self._rest.request(
            "GET", "/messages", params={"q": query, "maxResults": limit}
        )
        ids = [item["id"] for item in listing.get("messages", [])]
        return list(await asyncio.gather(*(self._fetch(id_, "metadata") for id_ in ids)))

    async def _fetch(self, message_id: str, fmt: str) -> EmailMessage:
        params: dict[str, Any] = {"format": fmt}
        if fmt == "metadata":
            params["metadataHeaders"] = list(_METADATA_HEADERS)
        data = await self._rest.request("GET", f"/messages/{message_id}", params=params)
        return parse_message(data)


def parse_message(data: dict[str, Any]) -> EmailMessage:
    """Map a Gmail ``Message`` resource to the neutral model."""
    payload = data.get("payload", {})
    headers = {h["name"].lower(): h["value"] for h in payload.get("headers", [])}
    received = datetime.fromtimestamp(int(data["internalDate"]) / _MS_PER_SECOND, tz=UTC)
    return EmailMessage(
        id=data["id"],
        thread_id=data.get("threadId", data["id"]),
        sender=headers.get("from", ""),
        recipients=tuple(part.strip() for part in headers.get("to", "").split(",") if part),
        subject=headers.get("subject", ""),
        snippet=data.get("snippet", ""),
        received_at=received,
        unread="UNREAD" in data.get("labelIds", []),
        body=_plain_text_body(payload),
        rfc822_id=headers.get("message-id"),
    )


def _plain_text_body(payload: dict[str, Any]) -> str:
    """First ``text/plain`` part, searched depth-first through multiparts."""
    if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
        return _decode_base64url(payload["body"]["data"])
    for part in payload.get("parts", []):
        text = _plain_text_body(part)
        if text:
            return text
    return ""


def _decode_base64url(data: str) -> str:
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")


def _encode_mime(mime: MimeMessage) -> str:
    return base64.urlsafe_b64encode(mime.as_bytes()).decode("ascii")


def _reply_subject(subject: str) -> str:
    return subject if subject.lower().startswith("re:") else f"Re: {subject}"


def _reply_mime(message: EmailMessage, body: str) -> MimeMessage:
    mime = MimeMessage()
    mime["To"] = message.sender
    mime["Subject"] = _reply_subject(message.subject)
    if message.rfc822_id:
        mime["In-Reply-To"] = message.rfc822_id
        mime["References"] = message.rfc822_id
    mime.set_content(body)
    return mime


# Design notes:
# - Listing is two round trips (ids, then one metadata fetch per id, run
#   concurrently). The batch endpoint would make it one, but for the ten or so
#   messages a voice summary covers the gain is not worth the multipart code.
# - Replies are built as RFC 5322 messages with the stdlib email package and
#   handed to Gmail base64url-encoded; threadId plus In-Reply-To keeps them in
#   the original conversation in Gmail and in other clients.
# - Only text/plain bodies are extracted. HTML-only mail falls back to the
#   snippet, which is what the agent reads aloud anyway.
