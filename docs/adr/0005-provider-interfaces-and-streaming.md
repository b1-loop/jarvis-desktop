# 0005. Provider interfaces, streaming speech and a third LLM provider

Date: 2026-10-21
Status: Accepted

## Context

Step 5 adds the first real speech-to-text, text-to-speech and LLM
providers. Three questions had to be settled before the agent loop (step 6)
and the voice pipeline (step 7) are written on top of them:

1. How are the stages shaped so that CLOUD and LOCAL mode, and the test
   suite, can swap implementations without touching the orchestrator?
2. How does speech start before the model has finished writing? A spoken
   assistant that waits for the whole answer feels several seconds slower
   than one that starts on the first sentence.
3. edge-tts returns MP3, and the audio engine plays float32 PCM. Something
   has to decode, and it has to do so while the bytes are still arriving.

A related product question: Claude is the quality default and llama.cpp is
the fully offline option, but many developers already run Ollama. Is it a
third provider or a distraction?

## Decision

### One interface per stage, provider-neutral types in `base.py`

`STT`, `TTS` and `LLMClient` are abstract base classes in their package's
`base.py`, together with the data they exchange (`Transcript`,
`AudioChunk`, the message/tool/event types). Providers convert to and from
their own wire formats; nothing provider-specific crosses the interface.
Each provider takes its external dependency through a narrow `Protocol`
(the faster-whisper model, the Anthropic streamer, the edge-tts chunk
stream, the MP3 decoder) so unit tests use scripted fakes and the only
network tests are marked `integration` and skip without credentials.

The one deliberate leak is `AssistantMessage.provider_state`: Claude's
signed thinking blocks must be replayed verbatim when a tool-use turn is
continued, so the client that produced a turn gets its own blocks back.
Other providers ignore the field.

### Streaming, sentence by sentence

`LLMClient.stream` yields text deltas as they arrive. `SentenceSplitter`
turns deltas into complete sentences; each sentence is handed to
`TTS.synthesize`, which itself yields PCM chunks as they are decoded, and
the `Speaker` plays chunks as they come. The first words are spoken while
the model is still writing the second sentence. The unit of work is a
sentence because it is the smallest piece that sounds natural when
synthesised on its own and it keeps at most one sentence of audio in
flight, which is what makes barge-in cheap to abandon.

### Native MP3 decoding with stateless batches and overlap priming

MP3 is decoded in C++ through miniaudio's built-in dr_mp3 (the audio
engine already ships miniaudio; enabling its MP3 decoder costs no new
dependency). The native API deliberately decodes one complete buffer with
no state between calls, which keeps it tiny and trivially testable. The
Python side makes that stream: a frame splitter parses Layer III headers
so only whole, confirmed frames are decoded, and every batch is primed with
the previous six frames so the decoder has the bit reservoir and
filterbank history the first new frame depends on. Only the output of the
new frames is kept. A `native`-marked test decodes a real edge-tts clip in
batches and compares it sample by sample with a whole-file decode.

### Ollama as a third LLM provider

`JARVIS_LLM_PROVIDER` selects `anthropic`, `ollama` or `local`. Ollama
speaks a small HTTP API with streaming and tool calling, needs only
`httpx`, and lets a developer run Jarvis without an API key on a model
they already have. It is not LOCAL mode: it still pairs with faster-whisper
and edge-tts, and it is an extra server to run. `local` (llama.cpp in
process, step 12) raises a clear `ConfigError` until it exists rather than
silently falling back.

Retries are one shared policy (`jarvis.utils.retry`) with exponential
backoff and jitter; SDK retries are switched off so that every provider
logs and behaves the same. Only establishing a stream is retried, because
text that has already been spoken cannot be taken back.

## Alternatives considered

- **Call provider SDKs directly from the agent** – fastest to write, but
  every provider swap and every test would touch the agent loop, and
  LOCAL mode would be a fork rather than a configuration.
- **Speak only when the answer is complete** – simpler pipeline, no
  sentence splitting, but adds the whole generation time to the perceived
  latency, which is the one number a voice assistant is judged on.
- **Decode MP3 in Python (`pydub`/ffmpeg, `miniaudio` wheel)** – an extra
  binary dependency or a subprocess per sentence; the project already
  compiles miniaudio, and keeping codecs in C++ matches the "C++ is the
  muscle" split.
- **A stateful native streaming decoder** – would avoid overlap priming,
  but dr_mp3 is pull-based and treats a short read as end of stream, so it
  cannot be paused to wait for the next network chunk; a frame-level
  decoder API would have meant exposing dr_mp3 internals.
- **Ask edge-tts for raw PCM** – the service only offers compressed
  formats for the neural voices.
- **Only Claude and llama.cpp** – fewer providers to maintain, but Ollama
  is the most common way developers run local models and costs about two
  hundred lines behind the existing interface.

## Consequences

- The agent loop and the voice pipeline are written once against three
  interfaces; CLOUD, LOCAL and Ollama are configuration.
- Perceived latency is bounded by first-token time plus one sentence of
  synthesis, not by the full answer.
- The MP3 path has two moving parts (frame splitter, batcher) that must
  agree with the decoder's behaviour; the native comparison test is the
  contract and must keep passing if miniaudio is upgraded.
- `provider_state` is a small provider-specific wart in a neutral type;
  it is documented and ignored by everyone except the Anthropic client.
- Three providers means three error vocabularies mapped onto `LLMError`
  with a remedy in the message, because the orchestrator reads it aloud.
