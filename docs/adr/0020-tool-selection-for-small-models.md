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
