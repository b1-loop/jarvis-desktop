# 0020. Per-turn tool selection and a grounding guard for small local models

Date: 2026-10-22
Status: Accepted

## Context

Ollama with `qwen2.5:7b` became the default LLM (ADR-0013, update of
2026-10-22). A real text session against it showed two failures that
Claude never produced:

1. **Tool overload.** With all 62 tool schemas in the request, "what time
   is it?" made the model call `shell_command` with `date`, saying no time
   tool existed. The identical request offering only `current_time`,
   `system_monitor` and `shell_command` produced a correct `current_time`
   call. A 7B model cannot pick reliably among sixty definitions that
   together are several thousand tokens of its context.
2. **Ungrounded success.** After the user declined the shell call, the
   model's final words were "Done. It printed: ansel", although nothing
   ran. The persona already said "never invent tool results"; the model
   did it anyway.

## Decision

**Select tools per turn.** `agent/tool_selection.py` defines a
`ToolSelector` Protocol and `RelevanceToolSelector`, which ranks every
registered tool against the user's words and offers the model:

- a small configurable core (`JARVIS_CORE_TOOLS`, default `current_time`,
  `web_search`), always;
- every tool already called earlier in the same turn, so chains keep their
  tools across rounds;
- the best-ranked remaining tools with a non-zero score, up to
  `Settings.tools_per_turn()`.

The cap defaults by provider: 0 (all tools) for Claude, 12 for Ollama and
llama.cpp, including LOCAL mode, which never uses Claude.
`JARVIS_MAX_TOOLS_PER_TURN` overrides it. The agent loop re-selects every
round and offers the result in registry order.

Ranking is deterministic word overlap: request words (lower-cased,
stopwords dropped, plural `s` stripped) that match a tool's name or its
new `keywords` class attribute score 3, matches in the description or
category score 1. `keywords` is filled for twelve tools where people's words
differ from the tool's (`current_time`: time, clock, date, day, today,
hour; `system_monitor`: ram, memory, cpu, battery, disk, slow; ...).

**Ground the closing words.** When every tool call of a turn failed or was
declined, the loop holds the model's next reply. If that reply calls tools
again, it is spoken. If it is the final answer, it is dropped (logged as
`agent.grounding.reply_replaced`) and the turn ends with the failed tools'
own `ToolResult.spoken` sentences, deduplicated, for example "Okay, I won't
do that." No summary request is made in that case. The persona also gains
one explicit sentence: a failed or declined tool did nothing, so never say
it is done.

## Measurements

35 labelled requests in `tests/agent/data/tool_selection_cases.yaml`
(phrasing that avoids tool names, such as "how much RAM am I using?" and
"make the screen dimmer"), against the real registry of 62 tools. A case
hits when an expected tool is among the top k.

| Ranker | top-1 | top-3 | top-5 | top-12 |
| --- | --- | --- | --- | --- |
| Word overlap, names and descriptions only | 0.80 | 0.91 | 0.97 | 0.97 |
| Word overlap plus `keywords` (chosen) | 0.89 | 1.00 | 1.00 | 1.00 |
| Memory's hashed bag-of-words embedding, cosine, with keywords | 0.49 | 0.69 | 0.77 | 0.86 |

What the model actually receives under the Ollama defaults (cap 12, two
core tools): the expected tool in 35 of 35 cases, 6.8 tools on average
instead of 62. `tests/agent/test_tool_selection.py` asserts top-1 >= 0.85,
top-3 >= 0.95 and a 100 % hit rate for the default selection.

The keywords were written with the labelled set in view, so the "with
keywords" row is optimistic; the cases are kept as a regression set and a
real session that hides the right tool should become a new case.

End to end against Ollama `qwen2.5:7b` (text mode, empty memory), "what time
is it?" now calls `current_time` and "how much RAM am I using?" calls
`system_monitor`; both answers matched the system clock and Windows'
memory counters. A declined `date` shell command ended with "Okay, I won't
do that." while the model's own closing text was dropped and logged.

## Alternatives considered

- **Reuse the memory module's hashing embedding.** Measured above: cosine
  over hashed word counts normalises by document length, so long, precise
  descriptions lose against short vague ones. It ranked worse at every k.
- **Ask the model to pick tools first.** Doubles latency per request and
  asks the same small model to make the decision it already gets wrong.
- **A real sentence-embedding model.** Better recall on paraphrases, but a
  new dependency and model download for a list of 62 short documents that
  word overlap already ranks perfectly at k=3.
- **Pad the list to the cap with unmatched tools.** Unrelated schemas are
  exactly what confuses small models; tools with score 0 are not offered.
- **Check the model's final text for success claims instead of replacing
  it.** Deciding whether free text claims success needs a classifier, and
  the failure case shows the model cannot be its own judge. Replacing is
  deterministic and the tools' sentences are short and true; the cost is
  losing a correct paraphrase or a follow-up question when every tool
  failed.

## Consequences

- Small models see 3 to 12 tools instead of 62; prompts shrink by several
  thousand tokens, which also shortens time to first token on CPU.
- A request whose words match no tool gets only the core set. The labelled
  set and a cap of 12 make that rare, and `JARVIS_MAX_TOOLS_PER_TURN=0`
  restores the full list for a capable local model.
- Requests are matched in English, like the tool descriptions. Another
  language falls back to the core tools.
- Tool authors should set `keywords` when users name the capability
  differently from its name and description; the docstring on `Tool` says
  so.
- Claude is unaffected by selection (cap 0) but does get the grounding
  guard, which costs it nothing when a tool succeeds.
- Not solved here: with an earlier session's "It is 09:11" in recalled
  memory, the 7B model repeated it a minute later without calling a tool.
  A prompt sentence calling remembered answers stale did not change that
  in repeated runs, so it was not kept. The fix belongs in how memory
  renders recent turns (timestamps, or leaving out answers derived from
  tools), which is outside this decision.

## Update 2026-10-22: multilingual selection, and thinking models

### Context

The consequence above ("another language falls back to the core tools")
turned out to be the everyday case: the user speaks Swedish
(`JARVIS_STT_LANGUAGE=sv`). Measured on this machine, `rank()` returned
nothing for "hur mycket batteri har jag?" and `close_app`, `search_email`,
`send_message` for "lägg till mjölk på att-göra-listan", so the 7B model
answered that it could not do that with the tools it had. Separately,
`qwen3:8b` answered RAM questions more reliably than `qwen2.5:7b` but was
about three times slower, because Ollama runs qwen3 with thinking on unless
told otherwise.

### Decision

**Hybrid ranking.** `RelevanceToolSelector` now blends two signals per tool,
each first scaled to [0, 1] for the request:

- word overlap (as before, now Unicode-aware with Swedish stopwords,
  hyphenated compounds split and joined, and a name or keyword at the
  start of a longer word scoring 2 instead of 3, for Swedish suffixes and
  compounds such as "batteriet" and "skärmdumpen"), divided by the best
  overlap of any tool;
- cosine similarity from a multilingual embedding model, scored 1 for the
  most similar tool and falling linearly to 0 at 0.08 below it.

The blend is `0.4 * words + 0.6 * similarity`
(`JARVIS_TOOL_SELECTION_KEYWORD_WEIGHT` / `_SEMANTIC_WEIGHT`). A tool whose
blended score is 0 is still never offered; the core set, the
already-used-this-turn rule, the cap and registry order are unchanged.

**Embeddings.** `agent/tool_embeddings.py` holds `EmbeddingToolRanker`,
which embeds each tool's spoken name, description and keywords once,
through Ollama's `/api/embed` (`llm/ollama_embeddings.py`, behind the
`llm.embeddings.TextEmbedder` Protocol), and keeps the vectors in
`data/cache/tool_embeddings.json`, keyed by a hash of model and text, so a
restart re-embeds nothing and an edited description or another model
re-embeds exactly what changed. The tools are embedded in the background at
startup; until that finishes, turns rank by words. Each request is embedded
per turn with a 150 ms budget (`JARVIS_TOOL_EMBEDDING_TIMEOUT_MS`): a late
embedding is not cancelled (it is usually Ollama loading the model) but
the turn goes on with words only. A missing model disables the ranker for
the session with one warning naming `ollama pull`; an unreachable server
backs off for 60 s, also logged once. `JARVIS_TOOL_EMBEDDING_MODEL=none`
turns it off.

**Model: `bge-m3`.** Pulled and compared on the same cases:

| Embedding model | Disk | VRAM loaded | Tools embedded | Query p50 / p90 |
| --- | --- | --- | --- | --- |
| `bge-m3` (chosen) | 1.2 GB | 689 MB | 2.6 s | 26 / 33 ms |
| `embeddinggemma` | 621 MB | 685 MB | 4.2 s | 24 / 35 ms |
| `nomic-embed-text` | 274 MB | not measured | 2.5 s | 20 / 32 ms |

`bge-m3` and `qwen3:8b` (5.6 GB) were loaded on the 8 GB GPU at the same
time (`ollama ps`). `nomic-embed-text` is English-only and ranked Swedish
poorly; `embeddinggemma` was close on the tuned cases but clearly worse on
held-out phrasing (below). Model-specific query/document prefixes
(`embeddinggemma`, `nomic-embed-text`) are applied by the embedder.

**Swedish keywords.** The common tools (time and date, battery, RAM/CPU,
volume, brightness, screenshot, lock, apps and windows, todos, notes,
reminders, calendar, email, web search, music, files, tests) gained Swedish
`keywords` next to the English ones. They are the baseline that still works
when the embedding model is unavailable.

**Thinking control.** `JARVIS_OLLAMA_THINK` (`Settings.ollama_think`,
`bool | None`, default `false`; `default` sends nothing and leaves it to the
model) is sent as Ollama's `think` field. Streamed `thinking` text is never
turned into answer text or spoken; only its length is logged, at debug.
With thinking off, `qwen3:8b` sometimes ends a reply at once with no text
and no tool call (three times in four on one request); the client asks once
more in that case, which is invisible to the caller because nothing was
streamed yet.

### Measurements

`tests/agent/data/tool_selection_cases.yaml` now has the 35 English cases,
36 Swedish cases written together with the Swedish keywords, and 28
held-out Swedish cases written afterwards and never used to write keywords.
The weights and the margin were chosen on all cases (a small grid over
weights 0.2 to 0.7 and margins 0.05 to 0.15, preferring the selected hit
rate, then fewer tools offered), so the hybrid rows are tuned on the
held-out set too; the keyword rows are not. Each cell is
top-1 / top-3 / top-12 / selected, where "selected" is what the model
receives under the Ollama defaults (cap 12, core `current_time`,
`web_search`), with the average number of tools offered in brackets.

| Ranker | English (35) | Swedish (36) | Swedish held-out (28) |
| --- | --- | --- | --- |
| Before: English word overlap (`main`) | 0.89 / 1.00 / 1.00 / 1.00 (6.8) | 0.06 / 0.06 / 0.06 / 0.14 (2.5) | 0.04 / 0.04 / 0.07 / 0.18 (3.0) |
| Word overlap + Swedish keywords | 0.89 / 1.00 / 1.00 / 1.00 (6.9) | 0.92 / 1.00 / 1.00 / 1.00 (4.3) | 0.46 / 0.57 / 0.57 / 0.57 (3.6) |
| Embedding only, `bge-m3` | 0.83 / 0.89 / 0.97 / 1.00 (12) | 0.75 / 0.89 / 1.00 / 1.00 (12) | 0.89 / 0.93 / 1.00 / 1.00 (12) |
| **Hybrid, `bge-m3` (chosen)** | 0.94 / 1.00 / 1.00 / 1.00 (8.2) | 0.89 / 1.00 / 1.00 / 1.00 (6.0) | 0.89 / 0.93 / 1.00 / 1.00 (7.1) |
| Embedding only, `embeddinggemma` | 0.89 / 0.94 / 1.00 / 1.00 (12) | 0.81 / 0.92 / 0.97 / 0.94 (12) | 0.71 / 0.89 / 1.00 / 1.00 (12) |
| Hybrid, `embeddinggemma` | 0.94 / 1.00 / 1.00 / 1.00 (7.6) | 0.89 / 1.00 / 1.00 / 1.00 (5.1) | 0.64 / 0.89 / 0.93 / 0.93 (5.8) |
| Embedding only, `nomic-embed-text` | 0.83 / 0.91 / 1.00 / 0.97 (12) | 0.47 / 0.61 / 0.89 / 0.86 (12) | 0.29 / 0.46 / 0.82 / 0.79 (12) |
| Hybrid, `nomic-embed-text` | 0.91 / 0.97 / 1.00 / 1.00 (8.1) | 0.72 / 0.94 / 1.00 / 1.00 (10.5) | 0.54 / 0.68 / 0.86 / 0.86 (11.2) |

"Embedding only, selected" is the top 10 by similarity plus the core set.
The held-out keyword row (0.57) is the honest estimate of what keywords
alone do for phrasing nobody anticipated; the embedding model is what
closes that gap. GPU embeddings vary slightly between runs (top-1 moved by
up to 0.03).

The pure tests keep the keyword thresholds for English and Swedish and
assert that the held-out keyword rate stays low (keywords were not tuned on
it). `tests/agent/test_tool_selection_ollama.py` (marked `integration`,
skipped without Ollama and `bge-m3`) asserts the hybrid rates.

End to end, `scripts/compare_models.py` sends seven Swedish requests
through the real agent loop and tool selection with a temporary
`data_dir`, the stub OS layer and every non-read-only tool replaced by one
that refuses to run, and checks the answers against `psutil`, the clock
and the todo and reminder store (RTX 4070 Laptop, 8 GB; models warm):

| Model | Correct | Median time to first token | Total per round of 7 |
| --- | --- | --- | --- |
| `qwen2.5:7b`, think off (3 rounds) | 15/21 | 1.3 s | 26.7 s |
| `qwen3:8b`, think off (3 rounds) | 18/21 | 1.5 s | 28.1 s |
| `qwen3:8b`, think on (2 rounds) | 14/14 | 13.7 s | 135.5 s |

Both small models now call the right tool for every Swedish request that
has one. `qwen2.5:7b` reported a single process's memory as "your RAM" in
two of three rounds; `qwen3:8b` reported RAM correctly every time. Every
`qwen3:8b` think-off miss, and three of `qwen2.5:7b`'s, is "påminn mig om fem
minuter": the model invents a date (2023) instead of calling `current_time`
first, and the tool rejects it as past. Thinking fixes that, at ten times
the latency. The RAM request's first token takes about 10 s for every
model because `system_monitor` itself samples processes for that long.

### Consequences

- Swedish requests reach the right tool; the measured failure cases are
  regression tests.
- A second model on the GPU: about 0.7 GB of VRAM while loaded, kept for
  30 minutes after use (`keep_alive`) so a quiet period does not cost a
  model load on the next request. Without `ollama pull bge-m3`, the
  selection is exactly the keyword ranking, with one warning.
- `ToolSelector.select` is now `async`.
- `qwen3:8b` with thinking off is the better local model on these numbers
  (more correct, same speed); thinking on is not usable for voice. The
  default model is not changed by this decision.
- Not solved here: relative reminders with thinking off. The fix is
  probably to give `add_reminder` an "in N minutes" field or to put the
  current local time in the system prompt; either is its own change.
