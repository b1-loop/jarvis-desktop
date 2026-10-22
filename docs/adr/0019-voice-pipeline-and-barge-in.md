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

## Update (2026-10-22): echo-safe barge-in, GPU speech recognition, language discipline

### Context

The first week on a real laptop (RTX 4070 Laptop, built-in speakers,
Swedish voice) showed three failures in one log:

- `speaking -> listening reason=barge-in` twice with nobody talking: the
  speakers' echo of Jarvis's own voice passed the 400 ms guard and the
  250 ms `minSpeechMs`, exactly the risk recorded above.
- "Vad finns på det?" for "Vad finns på Discord?": faster-whisper `small`
  on the CPU, because CTranslate2 found no cuBLAS 12 / cuDNN 9 DLLs and
  the provider fell back to the CPU.
- A reply that began in Swedish and continued in Chinese
  ("Var god och的具体任务..."), which the Swedish voice then read aloud.

### Decision

**Barge-in modes** (`JARVIS_BARGE_IN_MODE`, `core/barge_in_mode.py`).
`voice` keeps the behaviour above; `wake_word` lets only the wake word, the
hotkey or typing interrupt; `off` only the hotkey or typing; `auto` (the
default) picks `voice` when the default output device is headphones or a
headset and `wake_word` otherwise. The device is read once per session from
Core Audio's `PKEY_AudioEndpoint_FormFactor` through a small read-only
addition to `jarvis_os` (`default_output_endpoint()`), with the endpoint
and microphone names ("headphone", "headset", "hörlur") as the fallback
when the form factor is unknown or the native module is missing; unknown
means speakers. The decision is logged as `barge_in.mode`, and the greeting
says how to interrupt.

**Python watches while Jarvis speaks.** `BargeInController` now runs
watchers between the first audio of an answer and its end: the engine's
event queue (nobody else reads it then) for `WAKE_WORD_DETECTED` and, in
voice mode, `SPEECH_STARTED`, and the global hotkey, which previously was
not polled at all while Jarvis spoke. A wake word or hotkey stops playback,
cancels the turn and moves SPEAKING -> LISTENING (`barge-in (wake word)`);
the voice input then plays the earcon and captures a fresh request, as from
idle. The native target is armed only in voice mode with the echo check
disabled.

**Echo check for voice mode** (`core/echo_detector.py`). Microphone and
playback RMS are sampled every 10 ms while speaking. The echo path's
coupling (input/output ratio) is estimated as the lower quartile of the
ratios seen so far, which the echo guard window provides; an onset whose
mean input over the last 300 ms is below `ratio * coupling * output`
(`JARVIS_BARGE_IN_ECHO_RATIO`, default 2) is ignored as echo. Calibrating
instead of comparing raw levels matters: the playback level is digital and
the microphone level acoustic, so with headphones a fixed `input < k *
output` would reject the user's own voice.

**Self-wake guard.** While speaking, a wake word must score at least
`wake_word_threshold + JARVIS_SPEAKING_WAKE_MARGIN` (0.9 by default), and
is ignored for `JARVIS_SELF_WAKE_GUARD_MS` (2.5 s) after a sentence that
contains the assistant's name starts playing.

**GPU speech recognition without a CUDA install.** A new `gpu` extra pins
`nvidia-cublas-cu12==12.9.1.4` and `nvidia-cudnn-cu12==9.10.2.21` (Windows
and Linux). Before CTranslate2's CUDA backend loads, `stt/cuda_libraries.py`
adds the wheels' `bin` directories to the DLL search path
(`os.add_dll_directory` plus `PATH`). The ADR-0020-era CPU fallback stays
and also covers "no CUDA device". Models are chosen per device:
`JARVIS_STT_MODEL_GPU=large-v3-turbo`, `JARVIS_STT_MODEL_CPU=small`;
`JARVIS_STT_MODEL` still pins one model for every device. `auto` compute
becomes `int8_float16` on CUDA. `JARVIS_STT_HOTWORDS` (default "Jarvis,
Discord, Gmail, Spotify") is passed to faster-whisper's `hotwords`.

Measured on this machine (RTX 4070 Laptop 8 GB shared with Ollama, a 13.9 s
Swedish clip synthesised with edge-tts `sv-SE-MattiasNeural`, language
forced to `sv`, beam 5, warm runs):

| Model | Device / compute | Load (cached) | Transcribe (warm) | RTF | Word errors |
| --- | --- | --- | --- | --- | --- |
| large-v3-turbo | CUDA float16 | 4.3-7.2 s | 0.53-0.72 s | 0.04-0.05 | 1 ("Gale" for Gmail) |
| large-v3-turbo | CUDA int8_float16 | 4.9-5.8 s | 0.55-0.58 s | 0.04 | 1 ("Gail") |
| small | CPU int8 | 2.0-2.4 s | 2.6-2.7 s | 0.19 | 4 (mail, jass, Gale, Påminner) |

int8_float16 used about 1 GB less VRAM than float16 for the same speed and
accuracy, which is why it is the GPU default next to a 7B Ollama model.
Hotwords did not fix "Gmail" in this synthetic clip (the TTS voice
pronounces it the Swedish way); they are kept because the bias is cheap
and the names are exactly the ones users say.

**Language discipline** (`agent/persona.py`, `agent/language_guard.py`).
The persona names the language in English and in itself ("Always reply in
Swedish (svenska) ... never switch to Chinese or any other language"). The
agent loop drops every sentence containing CJK script when the configured
language is not Chinese, Japanese or Korean, and regenerates a final reply
in which that happened once, with the rule repeated in the system prompt;
the drifted reply never enters the history.

### Consequences

- On laptop speakers Jarvis can no longer interrupt itself; the price is
  saying "Hey Jarvis" (or the hotkey) to interrupt. Headphone users keep
  talk-over barge-in, now decided in Python (adds at most ~10 ms) unless
  `JARVIS_BARGE_IN_ECHO_RATIO=0` restores the native stop.
- `uv sync --extra gpu` adds about 1 GB of wheels; without it nothing
  changes except the CPU model choice. The first CUDA transcription after
  load takes 1-2 s (kernel initialisation); `warm_up` absorbs the load,
  not that first decode.
- AEC (WebRTC APM) remains the proper fix for speakers and stays the
  recorded follow-up; the energy check is a heuristic.
