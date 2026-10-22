# 0013. CLOUD vs LOCAL runtime modes

Date: 2026-10-22
Status: Accepted

## Context

Jarvis promises two runtime modes (`JARVIS_MODE`): CLOUD (Claude,
faster-whisper, edge-tts) and LOCAL (llama.cpp, whisper.cpp, Piper, fully
offline). ADR-0005 fixed the provider interfaces and added Ollama as a
third LLM option; ADR-0010 wrapped whisper.cpp and llama.cpp in C++. Step
12 had to decide how the Python side chooses a stack, how a model without
a native tool API calls tools, how Piper is run, and, above all, what a
user actually gives up or gains by switching. A voice assistant is judged
on time to first audio, so the decision needed numbers from real hardware,
not from model cards.

## Decision

### One switch, explicit prerequisites

`jarvis.core.providers` exposes `build_stt(settings)`, `build_tts(settings)`
and `build_llm(settings)`:

| Mode | STT | TTS | LLM |
| --- | --- | --- | --- |
| CLOUD | faster-whisper | edge-tts | `JARVIS_LLM_PROVIDER` (anthropic, ollama or local) |
| LOCAL | whisper.cpp | Piper | llama.cpp; Ollama only if chosen explicitly |

Every prerequisite is checked when the provider is built: optional extra
(`cloud`, `local`), native build (`jarvis_inference`, `jarvis_audio` for
edge-tts' MP3 decoding), model files, API key. A missing one raises
`ConfigError` naming the command that fixes it. Nothing falls back to
another provider silently. In LOCAL mode a cloud `llm_provider` (the
default, and what `.env.example` sets) is replaced by llama.cpp and the
switch is logged: `--mode local` must work on a checkout configured for
the cloud, and "nothing leaves the machine" outranks a provider default.

### LOCAL providers

- `WhisperCppSTT` and `LocalLlamaClient` drive `jarvis_inference` through
  Protocols (unit tests use fakes). Each blocking native call runs in
  `asyncio.to_thread`; tokens are pulled with one thread hop per
  `next_token()` rather than a reader thread feeding a queue, because the
  native `GenerationStream` already is a producer thread plus a queue.
  Cancelling the consumer (barge-in) cancels the native generation.
- **Tool calling by prompt.** The wrapper generates plain text, so the
  tool schemas are rendered into the system prompt and the model is asked
  to reply with exactly one `{"tool": name, "arguments": {...}}` object when
  it wants a tool. A streaming parser decides from the first non-blank
  characters whether the reply is prose (released immediately, so speech
  is not delayed) or a candidate object (held until its braces balance),
  tolerates the wrappers small models emit anyway (```` ```json ````,
  `<tool_call>`, `name` for `tool`, flattened arguments), validates the
  result shallowly against the tool's schema and otherwise falls back to
  text. The object is data only: a name looked up among the offered tools
  and arguments the tool validates again with pydantic. Generation stops
  as soon as a call validates (one call per turn; the agent loop asks
  again with the result, as with the cloud providers).
- **Piper via the `piper-tts` package.** Since 1.3 it ships abi3 wheels for
  Windows and manylinux that bundle espeak-ng and its data, so `uv sync
  --extra local` works on the dev machine and on CI with no compiler or
  system package; its one runtime dependency, onnxruntime, is already
  installed by the cloud extra. `PiperTTS` pulls `PiperVoice.synthesize`
  one sentence chunk at a time in a worker thread. The audio engine's own
  `onnxruntime.dll` and the wheel's coexist in one process (verified in
  both import orders).
- Pinned models (`download_models.py --local`, 630 MB): whisper `tiny`,
  Qwen2.5 0.5B Instruct Q4_K_M, Piper `en_GB-northern_english_male-medium`
  (CC-BY-SA 4.0 data, chosen over voices whose data is research-only or
  non-commercial). `Settings.local_*` default to these files; any other
  ggml / GGUF / Piper file is a setting away.

### Measured

`scripts/benchmark_local.py` (median of 3 runs after one warm-up; the
warm-up column includes model loading). STT input is the same 3 s clip of
Piper saying "Hey Jarvis, what is on my calendar for tomorrow morning?";
the LLM prompt asks for three sentences; TTS speaks one 60-character
sentence.

Hardware: AMD Ryzen AI 9 HX 370 w/ Radeon 890M (12 cores / 24 threads),
31 GB RAM, Windows 11 (build 26300), CPU only for every in-process model.

| Stage | Provider | Warm-up (incl. load) | Latency / time to first token / first audio | Tokens/s | Notes |
| --- | --- | --- | --- | --- | --- |
| STT (3 s audio) | faster-whisper `small` (cpu, int8, beam 5) | 6979 ms | 3443 ms | - | heard the sentence exactly |
| STT (3 s audio) | whisper.cpp `ggml-tiny.bin` (greedy) | 723 ms | 479 ms | - | heard the sentence exactly |
| LLM | Ollama `qwen3-coder:30b` (30B MoE) | 8810 ms | 348 ms | 16.2 | single run after warm-up; server was gone for the 3-run pass |
| LLM | llama.cpp `qwen2.5-0.5b-instruct-q4_k_m.gguf` | 1168 ms | 100 ms | 102.3 | |
| LLM | Claude | - | - | - | skipped: no `ANTHROPIC_API_KEY` on the benchmark machine |
| TTS (first audio) | edge-tts `en-GB-RyanNeural` | 1796 ms | 599 ms | - | network round trip to Microsoft |
| TTS (first audio) | Piper `en_GB-northern_english_male-medium` | 4680 ms | 128 ms | - | |

Reading the table: once loaded, LOCAL reaches first audio in roughly
0.5 s (STT) + 0.1 s (first token) + one sentence of generation + 0.13 s
(Piper), against 3.4 s + network TTFT + 0.6 s for CLOUD on the same CPU.
The price is load time at start-up (about 7 s, dominated by Piper's voice
and espeak data) and much weaker models. The faster-whisper row is slow
because `small` with beam search is six times the parameters of `tiny`
decoded greedily; on a CUDA machine it would be far faster. The Claude row
is not measured here and is the one number a reader must take from their
own run of the script.

## Trade-offs

| | CLOUD (Claude) | Ollama option | LOCAL (in process) |
| --- | --- | --- | --- |
| Latency | network round trips for LLM and TTS; best model, highest TTFT variance | LLM local, TTS still cloud | lowest time to first audio once loaded; seconds of model load at start |
| Quality | best reasoning and native tool use | good with a 7B-30B model, native tool API | 0.5B model: fluent but shallow, prompt-level tools are fragile; whisper tiny mishears more |
| Privacy | audio stays local (faster-whisper), text and answers go to Anthropic and Microsoft | text stays local, spoken answers go to Microsoft | nothing leaves the machine |
| Cost | per-token API cost | free, needs RAM/VRAM for the model | free; 630 MB of models, ~1 GB RAM |
| Setup | API key, cloud extra | Ollama server and a pulled model | native build, `local` extra, model download |

## Alternatives considered

- **Silent fallback** (LOCAL if the cloud is unreachable, edge-tts if Piper
  is missing) – convenient, but the user would never know which privacy
  and quality contract is in force. Degradation stays an explicit,
  spoken offer (ARCHITECTURE.md section 9).
- **GBNF grammar-constrained decoding for tool calls** – llama.cpp can force
  JSON, but forcing it for every reply would make prose impossible, and
  deciding up front whether a turn is a tool turn needs a second model
  call. It also means extending the C++ wrapper. Worth revisiting with a
  "tool turn" heuristic once the agent loop exists.
- **The model's own tool format (Qwen `<tool_call>`)** – works for one model
  family only; the JSON protocol works with any GGUF chat model, and the
  parser already accepts Qwen's habits.
- **Piper through onnxruntime + espeak-ng by hand** – would replace a pip
  install with our own phonemiser bindings for no gain. **Piper CLI
  subprocess** – adds process start-up to every sentence.
- **Ollama as LOCAL mode** – private and much better models, but an extra
  server to install and keep running, and no C++ in the loop; it stays the
  middle option.

## Consequences

- `--mode local` runs the whole pipeline offline after `uv sync --extra
  local`, `scripts/build_native.py` and `download_models.py --local`.
- `piper-tts` and espeak-ng are GPL-3.0-or-later. They are an optional
  extra, imported only at run time and never vendored, so Jarvis' own
  source stays MIT; anyone redistributing a bundle that includes the extra
  must honour the GPL.
- The pinned 0.5B model proves the code path, not the product: it
  sometimes calls a tool for a request that needs none. Pointing
  `JARVIS_LOCAL_LLM_MODEL` at a 3B-7B GGUF is the expected upgrade; the
  benchmark script quantifies the cost on the user's machine.
- whisper.cpp exposes no confidence, so LOCAL transcripts carry
  `confidence=None` and the "did you say ...?" heuristic stays CLOUD-only.
- Faster-whisper's `stt_device=auto` picks CUDA when CTranslate2 reports a
  device even if the CUDA runtime (`cublas64_12.dll`) is missing, and then
  fails to load; the benchmark pins `cpu`/`int8`. Choosing a safer CLOUD
  default is left to the CLOUD provider owner.
