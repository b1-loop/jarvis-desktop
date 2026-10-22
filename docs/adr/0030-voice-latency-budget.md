# 0030. Voice latency: measure each part, keep models warm, synthesise in parallel

Date: 2026-10-22
Status: Accepted

## Context

On the development laptop (RTX 4070 Laptop, 8 GB; Ollama `qwen2.5:7b` on
the GPU; faster-whisper `large-v3-turbo` on CUDA; edge-tts
`sv-SE-MattiasNeural`; Swedish speech) the live log showed about 8 s from
the end of speech to the first audio: 0.7 s silence hangover, 2.2 s
speech recognition, 2.4 s to the first LLM token (12 s once, after another
Ollama model had evicted `qwen2.5:7b`) and 1.0 to 4.5 s from the first
sentence to its audio, with audible gaps between sentences. The target is
under 3 s typical, with no gaps.

`scripts/benchmark_latency.py` measured the four stages but not their
parts, so it could not say whether the LLM stage was the model load, the
prompt or the generation, or whether the TTS stage was the network. It
also no longer ran: the voice front end had started passing
`wake_word_threshold` to the engine factory, which the benchmark's
factory did not accept.

## Decision

Measure first, then fix what the measurements point at.

**Measurement.** The benchmark taps the providers' own log events (a
structlog processor in front of the normal ones) and reports, per stage,
the STT decode, Ollama's `load_duration` and `prompt_eval_duration`, the
prompt size in tokens and the TTS provider's first-chunk time.
`--set KEY=VALUE` runs any variant (`stt_beam_size=5`,
`tts_provider=piper`, `vad_silence_end_ms=500`) without touching `.env`.
Its engine factory forwards every segmenter setting the application
passes, which also repairs the run.

**Fixes, each with the measurement behind it:**

| Change | Measured on this machine |
| --- | --- |
| `localhost` in `JARVIS_OLLAMA_BASE_URL` is rewritten to `127.0.0.1` (and that is the default) | Windows resolves `localhost` to `::1` first and Ollama listens on IPv4 only: +256 ms per new asyncio connection (Happy Eyeballs delay), +2.4 s per blocking one. httpx drops idle connections after 5 s, so nearly every turn paid it, for the chat and the embedding request. |
| Every Ollama request carries `keep_alive` (`JARVIS_OLLAMA_KEEP_ALIVE`, default `30m`) and an optional fixed `num_ctx` | Ollama's default unloads the model after 5 idle minutes; the next turn waited for a reload (seconds, 12 s in the user's log). A request with another `num_ctx` reloads it too. |
| The LLM is loaded at startup in the background (`JARVIS_LLM_WARM_UP`) | The first request no longer pays the load. |
| A turn whose `load_duration` exceeds 1 s logs `llm.model_was_not_loaded` | Makes eviction by another model visible in the log. |
| faster-whisper decodes greedily (`JARVIS_STT_BEAM_SIZE=1`), without timestamps or conditioning on earlier windows | 20 Swedish edge-tts commands, interleaved: same transcripts as beam 5; median 725 vs 762 ms, p90 1858 vs 2934 ms. |
| `warm_up` also decodes a second of silence | First decode after a load: about 2 s; afterwards 0.4 s. In the benchmark the first utterance went from 1914 ms to 389-467 ms. |
| `vad_filter` stays off | The native engine already cut the utterance with the same Silero model; the built-in pass added time (821 vs 762 ms) and changed nothing. |
| Each sentence is synthesised in its own task, at most `lookahead + 1` in flight | edge-tts's first byte is a service round trip (median 641 ms, p90 1773 ms; TLS is only 76 ms of it). Three sentences were complete after 2.7 s one after another and after 0.9 s concurrently. |
| `JARVIS_TTS_PROVIDER=piper` speaks with the local Piper voice in CLOUD mode; the Swedish voice `sv_SE-nst-medium` is in the manifest (`download_models.py --piper-sv`) | Piper's first chunk: median 97 ms, p90 118 ms (real-time factor 0.03, CPU), against edge-tts's 0.6 / 1.8 s. The voice loads (5 s) at startup now, not on the first answer. |
| Piper calls restore `LC_CTYPE` afterwards | espeak-ng switches the process to `en_US.UTF-8`; Python then decoded the Windows time-zone name ("Västeuropa, sommartid") with lone surrogates and the next LLM request failed to encode. Jarvis stopped on its second turn after speaking with Piper. |
| `JARVIS_VAD_SILENCE_END_MS` exposes the segmenter's hangover (default 700 ms) | 500 ms cut the wait after the last word by 200 ms in the benchmark (3022 vs 3465 ms median) with the same transcripts. |

Before and after, Swedish, cloud mode, null sink, median / p90 ms of six
measured turns (two prompts, three runs each, after one warm-up), on
`main` at be7cb4a and on this change:

| Stage | main, edge-tts | this change, edge-tts | this change, Piper |
| --- | ---: | ---: | ---: |
| End of speech -> transcript | 544 / 581 | 512 / 586 | 488 / 564 |
| Transcript -> first LLM token | 940 / 1340 | 791 / 1246 | 846 / 1225 |
| of which: prompt evaluation (Ollama) | - | 30 / 491 | 198 / 465 |
| First sentence -> first audio | 1356 / 2628 | 1564 / 3046 | 173 / 232 |
| **End of speech -> first audio** | **4057 / 4763** | **3772 / 4528** | **2727 / 3592** |
| Last word -> first audio (700 ms hangover) | 4791 / 5489 | 4510 / 5252 | 3465 / 4330 |

The steady-state benchmark hides what the user felt most: its warm-up
turn absorbs the first-request costs, and its turns come seconds apart, so
the model never expires. Those are fixed by the warm-ups and `keep_alive`
above, not visible in this table.

## Alternatives considered

- **A persistent edge-tts websocket.** The `edge-tts` package opens one
  connection per request and has no API to reuse it; connecting is 76 ms of
  a 641 ms first byte, so a hand-written client of an undocumented
  protocol would save about a tenth of the stage. Concurrent requests
  remove the gaps instead.
- **A smaller first MP3 batch.** After the first byte, edge-tts delivers
  11 times faster than real time; 20 frames arrive within 20 ms of the
  first, so batching costs nothing measurable.
- **Speculative first clause** (speak up to a comma after six words). Not
  done: with Piper the first sentence is audible in 0.2 s, and with edge-tts
  each extra request costs a full service round trip, so splitting a
  sentence trades one wait for two. It would also break the prosody of
  Swedish subordinate clauses.
- **A filler word ("Mm," / "Okej,").** Not done: it would mask, not
  remove, latency, and it would be spoken before the answer even when the
  answer is a one-word confirmation. Revisit only as an opt-in.
- **Switching the default TTS to Piper.** Not done: the user's voice is
  edge-tts's neural `MattiasNeural`, and `nst-medium` sounds clearly more
  synthetic. The numbers are reported; the choice stays with the user.
- **Moving the clock and recalled memory out of the system prompt.**
  qwen2.5's chat template renders the tool definitions (about 1000 tokens)
  *after* the system text, so anything volatile in the system prompt
  invalidates Ollama's prefix cache for the tools as well: a 3200-token
  prompt evaluated in 1.5 s on a miss and 36 ms on a hit. Moving the clock
  section to the end of the system prompt gains nothing (the tools still
  follow it); moving the clock and the memory into the latest user message
  would let the cache cover the rules and the tools. That changes what the
  grounding work (ADR-0029) tuned and evaluated, so it is left as a
  measured recommendation rather than done blind.

## Consequences

- With the local Piper voice the typical wait is under 3 s on this machine;
  with edge-tts it is set by the network (first byte p90 1.8 s, some
  runs several seconds).
- On an 8 GB GPU, `qwen2.5:7b` (4.75 GB), `bge-m3` (0.7 GB), Whisper
  (about 1 GB) and the desktop fill the card. Another Ollama model evicts
  Jarvis's model (the 12 s turn), and a second Whisper instance (a benchmark
  next to a running Jarvis) slowed prompt evaluation from about 2100 to 640
  tokens per second. README, "When answers are slow", says so.
- On battery the laptop caps the GPU at 28 W; STT then varied between 0.4
  and 4 s for the same clip. Benchmarks are only comparable on mains power.
- `JARVIS_STT_LANGUAGE` matters: auto-detection runs the encoder a second
  time (median 1576 vs 762 ms on the Swedish clips).
- Tests never warm a real Ollama model: `JARVIS_LLM_WARM_UP` is off in the
  test environment like the other desktop side effects.
