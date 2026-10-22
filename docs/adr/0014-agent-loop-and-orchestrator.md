# 0014. Agent loop, task chaining and one orchestrator for text and voice

Date: 2026-10-22
Status: Accepted

## Context

Step 6b turns the pieces from steps 4, 5, 6a and 9 (platform, providers,
tool framework, memory) into a working assistant: `uv run jarvis
--text-only` holds a real conversation with tools and memory. Four
questions shaped the design:

1. Where does the answer get cut up for speech, and when?
2. Where is the "ask before acting" rule enforced now that a model, not a
   developer, chooses which tools to call and with which arguments?
3. How is a model that keeps calling tools (a loop, a misunderstanding, a
   prompt injection in a web page) stopped without breaking real
   multi-step requests such as "book Anna on Thursday, send her an invite
   and add the agenda to my notes"?
4. Text mode exists now and voice mode arrives in step 7. Are they two
   programs or one?

## Decision

**The agent loop streams sentences.** `agent/loop.py` feeds every text
delta through `SentenceSplitter` and hands each complete sentence to an
injected `SentenceSink` immediately, including sentences the model says
before it calls a tool ("Let me check Thursday."). The terminal prints
them; in step 7 the TTS stage will synthesise them. Perceived latency is
first-sentence time, which the loop publishes as an `agent.first_sentence`
`LatencySample` next to `agent.llm_round` and `agent.turn`.

**Confirmation stays central.** Every tool call, sequential or parallel,
goes through `ConfirmationPolicy.execute_with_policy` (ADR-0006). The
system prompt tells the model that the system asks the user before
destructive actions, so the model should just call the tool; it never
asks "shall I?" itself. A declined call is an ordinary `is_error` tool
result the model can explain.

**Chaining is bounded per turn.** `agent/chaining.py` adds a
`ToolChain` per request with:

- a tool-round limit (`JARVIS_MAX_TOOL_ROUNDS`, default 6) enforced by the
  loop; calls still pending when it is hit are answered with an error
  result, and the model is asked for a summary;
- a tool-call budget (16 executions per request); extra calls are refused
  with an error result;
- parallel execution of the calls one message requests, through
  `asyncio.gather` behind a semaphore (4 at once), with the confirmation
  callback behind a lock so prompts never interleave;
- de-duplication of identical calls (name plus canonical arguments) within
  the turn: the duplicate gets the original result instead of running
  again;
- a `TurnReport` (tools used, rounds, durations, failures, limits hit) for
  the dashboard and for the memory record of the turn.

A turn that used tools always ends with a spoken summary: if the model
falls silent after its tools, the loop asks once for a one-sentence
summary, and if that is empty too it speaks the tools' own `spoken`
results. LLM failures (after the providers' own retries) are spoken as an
apology with the provider's remedy text, plus a hint to switch to Ollama
or LOCAL mode when the provider is a cloud service.

**One orchestrator for every front end.** `core/orchestrator.py` owns the
state machine and the turn lifecycle against two Protocols,
`InputSource` and `OutputSink`. Text mode plugs in `TerminalInput` (a
daemon reader thread, so Ctrl+C and "exit" never wait for a blocked
`input()`), `TerminalOutput` and `TerminalConfirmation`. The turn runs as
its own task so `cancel_current_turn()` (barge-in, step 7) stops streaming
and tool execution at the next await. Typed input walks the same states as
speech (`IDLE -> LISTENING -> THINKING -> SPEAKING -> IDLE`) because the
transition table has no shortcut, so the overlay and dashboard behave
identically in both modes.

`core/bootstrap.py` is the composition root: small `build_*` functions,
one `AsyncExitStack` that closes everything in reverse order, and
documented fallbacks (`StubPlatform` without the native build,
`InMemoryVectorStore` without ChromaDB) that are logged as warnings.

## Alternatives considered

- **Return the whole answer, split it in the TTS stage** – simpler loop,
  but speech would wait for the full generation including every tool
  round, adding seconds to every request with tools.
- **Stream raw deltas to the sink** – lowest latency on paper, but TTS
  cannot speak half a word, so every consumer would reimplement sentence
  splitting.
- **Let the model ask for confirmation in conversation** – no special
  machinery, but it relies on the model remembering to ask, and a prompt
  injection can tell it not to. The policy cannot be talked out of asking.
- **A single global limit on LLM requests** – easy, but it cannot tell a
  legitimate five-tool chain from a loop that calls the same tool five
  times. Rounds, calls and duplicates are separate signals and each has
  its own bound.
- **Run parallel calls one by one** – safer for prompts, slower for the
  common "look up A and B" request. Serialising only the confirmation
  prompt keeps both properties.
- **A separate text-mode REPL** – quick to write, but step 7 would then
  build a second turn lifecycle with its own state handling, and the two
  would drift. One orchestrator with swappable input and output avoids
  that.

## Consequences

- The voice pipeline only adds an `InputSource` (wake word, VAD, STT) and
  an `OutputSink` (TTS, playback) plus barge-in calling
  `cancel_current_turn()`; the agent and orchestrator stay as they are.
- Limits are per request. A user who really wants twenty actions in one
  breath gets a summary of what was done and what is left, by design.
- De-duplication means a user who asks for the same action twice in one
  request gets it once. Across requests nothing is cached.
- Conversation continuity comes from memory recall in the system prompt
  (ADR-0008), not from an in-process history, so a new session continues
  where the last one ended; the cost is that each turn's context is only
  as good as `MemoryManager.recall`.
- A cancelled turn is not recorded in memory because it has no final
  answer.
