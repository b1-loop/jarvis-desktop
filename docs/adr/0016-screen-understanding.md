# 0016. Screen understanding: downscale locally, byte budget, ask only on request

Date: 2026-10-22
Status: Accepted

## Context

Phase 2 asks for "screenshot → vision model": "what does this error mean?",
"summarise what I'm looking at", "what's the total in this sheet?". Four
questions sit underneath:

1. **Image size.** A 4K desktop is 8.3 megapixels and a multi-megabyte PNG.
   Claude bills images by pixel area (one visual token per 28×28 patch) and
   downscales anything above its native resolution itself: 2576 px on the
   long edge (4784 visual tokens) for the high-resolution tier (Claude 4.7
   and later, including the default `claude-sonnet-5-5`), 1568 px for older
   models. Pixels above that cost upload time and give nothing back.
2. **Byte limits.** Providers cap each image by its base64 length: 10 MB on
   the Claude API directly, 5 MB on Bedrock and Vertex. Accepted formats are
   PNG, JPEG, GIF and WebP.
3. **Privacy.** A screenshot is the most sensitive thing Jarvis can send:
   mail, chats, bank pages, passwords in plain text.
4. **Provider support.** Claude sees images; Ollama does only with a vision
   model (llava, qwen2.5vl, llama3.2-vision, gemma3); the default Ollama chat
   model and the llama.cpp wrapper do not.

## Decision

**A `ScreenReader` service in `vision/`, used by three read-only tools.**
`ScreenReader.ask(question, target)` captures through `Platform`
(`primary`, `all_monitors`, or `active_window` via
`Platform.screenshot_window`), passes the capture through a `Redactor`,
prepares it in a worker thread, and sends one user message (image first,
then the question, as the Claude docs recommend) with a short system prompt
to an injected vision `LLMClient`. It returns a `ScreenAnswer` with the text,
the size and format sent, latency, provider and the path of the kept
original. `describe_screen`, `explain_screen_error` and `ask_about_screen`
build the prompt and share `tools/_screen.py` for error phrasing.

**Downscale locally, then fit a byte budget.** The long edge is cut to
`JARVIS_VISION_MAX_LONG_EDGE` (default 1920: the 1080p class Anthropic
names as the balance of cost and accuracy for screenshots; 2576 is the
useful maximum on Claude, and Ollama vision models work at lower sizes
anyway). The image is encoded as PNG; if its base64 is above
`JARVIS_VISION_MAX_IMAGE_BYTES` (default 5 000 000, the strictest Claude
platform limit), JPEG at quality 90, 80, 70 follows, then the long edge
shrinks by a quarter and the round repeats, down to 512 px, below which the
request fails with a configuration hint instead of sending an unreadable
thumbnail. A real 2560×1600 desktop becomes a 1920×1200 PNG of about
0.98 MB base64 in under 200 ms.

**Privacy stance.** No background or periodic capture exists: the only
caller of `ask` is a tool, and a tool runs because the user asked. That
request is the consent, so the tools need no extra confirmation, but every
spoken answer from a cloud provider ends with "I sent a screenshot to
Anthropic to answer that." `JARVIS_VISION_ENABLED=false` refuses every
capture before the platform is touched. `active_window` never captures the
rest of the desktop. The `Redactor` Protocol (no-op `NoRedaction` today)
receives the full-resolution capture before downscaling and is the single
place to blur password fields or notifications later. The system prompt
tells the model not to read secrets aloud. Logs carry sizes, format, model
and latency only, never the image, the question or the answer. The
unredacted original is kept under `data/screenshots` so the user can see
what Jarvis looked at (`JARVIS_VISION_SAVE_SCREENSHOTS=false` keeps
nothing); it never leaves the machine.

**Provider support.** `vision/factory.py` builds the vision client through
`create_llm_client` with the model swapped: Claude with
`JARVIS_VISION_MODEL`; Ollama only when `JARVIS_OLLAMA_VISION_MODEL` is set
(the Ollama client already sends `images` on user messages); llama.cpp
never. Without a vision client the tools answer "I can't look at your
screen: the local model cannot see images; set JARVIS_OLLAMA_VISION_MODEL
…", and the same graceful path covers a missing platform backend, a
machine without screenshot support and a missing Pillow.

## Alternatives considered

- **Send the raw capture and let the provider resize.** Simplest, but a 4K
  PNG can exceed the 5 MB limit, uploads several times more data, and
  Ollama models do not resize sensibly on their own.
- **Always JPEG.** Smaller for photos, but screenshots are mostly text and
  flat colour where PNG is both smaller and lossless; JPEG artefacts are
  exactly what makes small text unreadable. JPEG is the fallback, not the
  default.
- **Confirm every upload ("send a screenshot to Anthropic?").** Doubles the
  round trips for a request the user just made. The spoken disclosure plus
  the master switch give the same transparency without the friction.
- **Crop the active window out of a full screenshot.** Would capture the
  whole desktop to keep one window, and `WindowInfo` carries no rectangle;
  the platform's window capture is both simpler and more private.
- **Reuse the chat `LLMClient`.** The vision model often differs from the
  chat model (always for Ollama), and a screen question is a one-shot
  request outside the conversation history.

## Consequences

- Screen questions cost a bounded, predictable number of tokens (about 2 700
  visual tokens at 1920×1080 on Claude) and never hit per-image limits.
- Pillow joins the code path; it is imported lazily and its absence is a
  clear spoken message, not an import error at startup.
- A per-monitor target needs a `Platform` method first; today the targets
  are primary, whole desktop and active window.
- Redaction is a seam, not a feature: until a detector lands, privacy rests
  on "only when asked", the disclosure and the master switch.
- The composition root has to build the reader with `create_screen_reader`
  and put it in `ToolContext.vision`; until then the tools say vision is not
  set up.
