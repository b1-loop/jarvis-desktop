# 0019. Voice pipeline: sentence pipelining, native-first barge-in, honest degradation

Date: 2026-10-22
Status: Accepted

## Context

Step 7 turns the text-mode assistant (ADR-0014) into a voice assistant:
wake word or hotkey, speech in, a streamed spoken answer out, and the user
able to talk over Jarvis and be heard at once. The pieces exist: the C++
audio engine with VAD, wake word, playback and a native barge-in hook
(ADR-0003), the Platform hotkeys (ADR-0004), streaming STT/TTS providers
(ADR-0005), the overlay (ADR-0011) and the orchestrator with its
`InputSource`/`OutputSink` seam and `cancel_current_turn()` (ADR-0014).
The open questions were:

1. How does the first word reach the speaker as early as possible when an
   answer is generated sentence by sentence and each sentence is
   synthesised over the network?
2. Who stops the audio when the user interrupts, and how fast?
3. Without acoustic echo cancellation, how is Jarvis kept from
   interrupting itself when its own voice reaches the microphone?
4. What happens when a part of the voice stack is missing on this machine?

## Decision

**One orchestrator, a voice front end.** `core/voice_io.py` implements the
orchestrator's Protocols: `VoiceInput` (trigger -> earcon -> LISTENING ->
`record_utterance` -> STT -> `TranscriptReady`), `VoiceOutput` (TTS ->
`Speaker`), and `VoiceConfirmation` (the question spoken, the yes/no heard).
`core/voice_bootstrap.py` builds them from `Settings`; `core/bootstrap.py`
gained a front-end builder seam so the rest of the assembly is shared with
text mode. `OutputSink` gained `flush()`, awaited inside the turn task, so
SPEAKING ends when the last word was played, not when the model stopped
writing.

**Sentence pipelining.** `VoiceOutput.say()` only queues the sentence. A
synthesiser task turns sentences into per-sentence chunk queues and a
player task feeds them to the `Speaker` in order, connected by a hand-off
queue bounded at two sentences. Synthesis of sentence *n + 1* overlaps
playback of sentence *n*, and edge-tts' own chunk streaming (MP3 decoded
natively in batches) means sentence one starts playing before it is fully
downloaded. Every sentence is also printed (`Jarvis> ...`) when it starts
playing, which keeps the terminal a transcript.

**Native-first barge-in, Python cleanup.** The engine stops the playback on
the VAD's `SpeechStarted` itself, inside one device period, when a
barge-in target is set (ADR-0003). `core/interruption.py`'s
`BargeInController` decides when that target is armed and handles the
aftermath: the voice output notices the stop (stop counter checked per
chunk and every 20 ms while waiting for audio), drops every queued
sentence and cancels synthesis, and the controller then (in this order)
starts retaining microphone audio, cancels the agent turn, and moves the
state machine SPEAKING -> LISTENING. The next `read_line` skips the trigger
and only waits for the end of the speech already in progress.

**Echo handling: debounce, not cancellation.** Two filters, both cheap:

- speech must persist for `JARVIS_VAD_MIN_SPEECH_MS` (250 ms) before the
  segmenter reports it at all (native `minSpeechMs`);
- the target is armed only `JARVIS_BARGE_IN_ECHO_GUARD_MS` (400 ms) after
  the first audio of an answer reaches the device, so the onset burst of
  Jarvis's own voice cannot stop it. The guard is implemented by arming
  late, because the native stop never consults Python.

Barge-in is held off entirely while Jarvis asks a confirmation question or
says "Sorry, I didn't catch that", because the user is expected to answer
right after (or over) those. There is no acoustic echo cancellation: with
loud speakers the echo can still trigger a barge-in after the guard. The
README recommends headphones or a lower volume.

**Degradation ladder, never faked.**

| Missing | Behaviour |
| --- | --- |
| `jarvis_audio` build, VAD model, microphone or speakers, or a speech provider prerequisite | `VoiceUnavailableError`; `uv run jarvis` says why and continues in text mode |
| wake-word models missing or rejected | warning; greeting says the wake word is unavailable and the hotkey works; hotkey-only |
| hotkey taken by another application | warning; wake word only (if neither works: text mode) |
| TTS fails for a sentence | the sentence is printed (as always), a one-time notice, the next sentence tries TTS again |
| STT fails for an utterance | logged with traceback, "Sorry, I didn't catch that." |

**Latency is measured per stage** (`core/voice_latency.py`) and published
as `LatencySample` events: `voice.wake_to_listening`,
`voice.speech_end_to_transcript`, `voice.transcript_to_first_token` (an
`LLMClient` decorator marks the first model event after a transcript) and
`voice.first_sentence_to_audio`. The dashboard's latency view and the
step-14 benchmark read them from the bus.

## Alternatives considered

- **Synthesise the whole answer, then play it** - simplest, but the user
  waits for the slowest sentence of the longest answer; rejected by the
  latency goal.
- **One continuous `Speaker.play` over all sentences** - gapless, but
  interruption and per-sentence printing/overlay updates would need a
  second protocol inside the chunk stream; per-sentence play with a
  pre-filled queue has no audible gap in practice.
- **Barge-in decided in Python** (consume VAD events, call
  `Playback.stop()`) - adds the event-loop poll interval and any loop load
  to the reaction time, which is exactly when the loop is busiest
  (streaming LLM, TTS). The native path does not depend on Python.
- **Acoustic echo cancellation (WebRTC APM, Speex AEC)** - the right fix
  for speakers, but it needs the playback signal as a reference inside the
  capture callback and careful clock alignment; out of scope for step 7
  and recorded as a follow-up. The guard plus `minSpeechMs` handle the
  headphone case and most moderate-volume speaker setups.
- **Ignore every VAD event during the first N ms instead of arming late** -
  impossible without a Python round trip per event, see above.

## Consequences

- Time to first audio is bounded by the first sentence: LLM time to the
  first sentence boundary plus TTS first-chunk time, not the whole answer.
- An interrupted answer wastes at most the two sentences synthesised
  ahead; the cancelled synthesiser closes the provider's stream.
- The orchestrator stayed unchanged except for `flush()`, a greeting, and
  not re-entering LISTENING when a voice source already did. Text mode
  walks the same states and uses the same agent.
- Without AEC, speaker users may see false barge-ins at high volume; the
  two settings above are the tuning knobs, and the overlay's waveform shows
  the input level while listening to help diagnose it.
- Proactive notices (ADR-0007) speak through the same `VoiceOutput`, but
  only when the assistant is IDLE, so they never interleave with a turn.
