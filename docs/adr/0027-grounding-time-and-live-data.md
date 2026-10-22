# 0027. Grounding small models: the clock in the prompt, relative reminders, no tool answers in memory

Date: 2026-10-22
Status: Accepted

## Context

ADR-0020 left two failures of the small local models open, both seen in
everyday Swedish voice use and both reproduced by
`scripts/compare_models.py`:

1. **Relative reminders.** "påminn mig om fem minuter att ringa mamma"
   failed with `qwen2.5:7b` and with `qwen3:8b` (thinking off) in every
   run. Neither model called `current_time` first; both wrote an absolute
   `due_at` with a year from their training data
   (`2023-10-22T18:05:00+03:00`) or an ISO duration (`P1M`) in the
   datetime field. `add_reminder` refused the past time, and the model
   then gave up instead of trying again, because the error said only "that
   time has already passed".
2. **Answers from memory.** Asked "vad står på min att-göra-lista?",
   `qwen2.5:7b` called no tool and spoke memory's own annotation, "(answered
   using list_todos; that result is out of date now, call the tool again
   for current values)", word for word. The memory layer had replaced old
   tool-derived answers in the recent conversation with that note, because
   showing the old answer made the model repeat the old data; the note was
   then repeated instead. Semantic hits still carried old tool answers in
   full ("User: ... Jarvis: <the list>").

## Decision

Ground the model in the present on every turn, give it a way to express
relative times that needs no arithmetic, and keep stale tool output out of
the prompt entirely.

- **The current moment is in the system prompt.** `Persona.system_prompt`
  takes `now` (from the turn's injected `Clock`) and states weekday, date,
  time to the minute, UTC offset and zone name, with the instruction to
  work relative dates out from it. It sits after the fixed rules and
  before memory.
- **`add_reminder` takes `in_minutes` or `due_at`, exactly one.** A
  pydantic validator rejects neither or both; `in_minutes` is bounded to
  1 minute .. 5 years and resolved against the clock in the tool. A
  `due_at` in the past, or more than five years ahead, is still refused
  and never stored; the error the model reads now states the current local
  time, names a wrong year explicitly, and points at `in_minutes`.
- **Tool-derived answers never reach the prompt as content.** In the
  recent conversation an assistant reply that called tools is left out;
  the user's question stays, and the section header says why answers are
  missing. Semantic hits for such turns are cut to their question half.
  Which turns used tools is decided at recall time from the stored
  message (`SqliteStore.tool_answer_ids`), so turns embedded before this
  change are covered too, and embeddings are unchanged.
- **A live-data rule in the prompt.** Todos, reminders, notes, calendar,
  email, messages, battery, CPU and memory use, music and windows must be
  fetched with a tool call in this turn, never answered from memory; and
  nothing counts as done (a reminder set, a todo added) unless a tool call
  in this turn did it. The memory section is labelled as background, not
  as an answer source.

## Alternatives considered

- **Only the clock in the prompt.** Leaves the model doing date arithmetic
  and writing ISO strings, which is where `P1M` came from; `in_minutes: 5`
  needs neither. Kept as well, because absolute times ("at six", "next
  Friday") still need the date.
- **Only `in_minutes`.** Fixes "in five minutes" but not "tomorrow at
  nine", where the model still had no idea of today.
- **An ISO 8601 duration field (`PT5M`).** The models already produced
  malformed durations in the datetime field; an integer is the format
  they get right.
- **Accepting a past time by rolling it forward a year or a day.** Guesses
  what the user meant and stores a reminder that may fire at the wrong
  time; asking back is safer.
- **Keeping a shorter note in place of old tool answers** ("used
  list_todos"). Anything in the assistant slot is a template the model
  can copy; an empty slot with the reason in the header cannot be spoken.
- **Not embedding tool turns at all.** Loses "what did we talk about"
  recall for half of all turns; cutting the answer at recall time keeps
  the question and can be revised without re-embedding.

## Consequences

`scripts/compare_models.py` gained two cases: the reminder request above
(its check now matches the reminder's text, so the two reminder cases
cannot pass on each other's row), and "list todos again", which adds a
todo straight to the store before asking a second time, so only a fresh
`list_todos` call can be right. Same machine as ADR-0020, models warm,
nine requests per round:

RESULTS_TABLE

- The system prompt changes once a minute, so a provider-side prompt cache
  covers the fixed rules before the clock section, not the whole prompt.
- An answer the model once gave from memory without a tool call is not
  tool-derived and still appears in the recent conversation; the prompt
  rule, not memory, guards against repeating it.
- "What did we talk about" still sees every question, but no longer the
  data a tool answered with (yesterday's agenda, last hour's battery).
  Facts about the user are stored separately and unaffected.
- The live-data list in the prompt must grow with new kinds of live tools.
