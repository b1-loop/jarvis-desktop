# 0028. Grounding small models: the clock, relative reminders, no tool answers in memory, the profile's name

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
3. **A name from history.** A speech-recognition error early in a session
   ("Hello, Jervis. I'm speaking with Ravel.") was stored as a turn.
   `qwen2.5:7b` then called the user Ravel in every reply, although the
   profile said otherwise, and each reply stored the wrong name again, so
   the history reinforced itself. The profile was only one line among the
   recalled sections, below nothing that marked it as more reliable than
   the transcript.

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
  recent conversation an exchange whose answer called tools is left out
  whole, question and answer, and the section header says why. Semantic
  hits for such turns are cut to their question half.
  Which turns used tools is decided at recall time from the stored
  message (`SqliteStore.tool_answer_ids`), so turns embedded before this
  change are covered too, and embeddings are unchanged.
- **A live-data rule in the prompt.** Todos, reminders, notes, calendar,
  email, messages, battery, CPU and memory use, music and windows must be
  fetched with a tool call in this turn, never answered from memory; and
  nothing counts as done (a reminder set, a todo added) unless a tool call
  in this turn did it. The memory section is labelled as background, not
  as an answer source.
- **Identity comes from the profile, not from history.** `RecalledContext`
  carries the profile's name (`user_name`) apart from its rendering, and
  the persona states it in its own section above all recalled text: this
  is the user's name, address them so, and any other name in the history
  is a recognition error. The recent-conversation header frames the turns
  as history only, not instructions, whose names and claims never override
  the profile, and each recalled assistant line is capped at 150
  characters so the model's own past words, the weakest evidence in the
  prompt, carry little weight.

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
- **Keeping the user's question and removing only the answer.** Measured
  worse (see below): a transcript of unanswered requests invites the
  model to answer in prose without tools.
- **Summarising recalled assistant replies with the LLM.** Would shrink
  the model's past words further, at the cost of a model call per turn on
  the latency path; a character cap does most of that for free.
- **Rewriting a wrong name in stored turns.** Requires knowing which word
  is a misheard name; stating the right one with authority does not.
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

| Model (think off) | Before | After |
| --- | --- | --- |
| `qwen2.5:7b`, all requests | 9/18 (2 rounds) | 27/27 (3 rounds) |
| `qwen3:8b`, all requests | 13/18 (2 rounds) | 26/27 (3 rounds) |
| both, the two reminder requests | 0/8 | 11/12 |
| `qwen2.5:7b`, "list todos" and "list todos again" | 0/4 (spoke memory's note) | 6/6 |

"Before" is `main` at PR #37; "after" includes PR #38 (system monitor
speed, ADR-0026), which is what fixed `qwen2.5:7b`'s RAM answers. The one
`qwen3:8b` miss is a correct `add_reminder(in_minutes=5)` that the Ollama
server only answered after a 600 s stall, so the reminder was due ten
minutes after the request; a shared server, not the model.

On the way there, keeping the user's questions of tool-answered exchanges
in the recent conversation (only the answers removed) scored 20/27 for
`qwen2.5:7b`: with a run of unanswered requests in the transcript, it
"answered" later requests in prose without calling tools ("Påminner om att
ringa mamma i fem minuter"). Dropping those exchanges whole scored 24/27
on the same code before PR #38 (the misses were RAM and one invented
todo list), so that is what memory does.

- The system prompt changes once a minute, so a provider-side prompt cache
  covers the fixed rules before the clock section, not the whole prompt.
- An answer the model once gave from memory without a tool call is not
  tool-derived and still appears in the recent conversation; the prompt
  rule, not memory, guards against repeating it.
- The recent conversation no longer shows tool-answered exchanges at all,
  and semantic recall finds their questions but not the data a tool
  answered with (yesterday's agenda, last hour's battery). Facts about the
  user are stored separately and unaffected.
- The live-data list in the prompt must grow with new kinds of live tools.
