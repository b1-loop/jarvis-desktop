# 0008. SQLite for structured memory, ChromaDB for semantic memory

Date: 2026-10-21
Status: Accepted

## Context

Jarvis has to remember things between sessions: who the user is, what they
work on, what was discussed, what is due and what is done. Two kinds of
question are asked of that memory and they want different indexes:

- **Exact questions**: "what reminders are due before 10:00?", "which
  todos are open?", "what is the user's name?". These need ordering,
  filters and transactions.
- **Fuzzy questions**: "have we talked about anything like this before?".
  These need nearest-neighbour search over embeddings and tolerate
  approximate answers.

The project is a desktop assistant, so everything must run locally with no
server to administer, and the test-suite must run without downloading a
model.

## Decision

Two stores behind one facade, `jarvis.memory.MemoryManager`:

- **SQLite** (via `aiosqlite`, WAL mode) is the source of truth for
  conversations, messages, notes, reminders, todos and **facts**. The schema
  is versioned with an ordered list of migration functions and a
  `schema_version` table. Full-text search uses FTS5 when the SQLite build
  has it and degrades to `LIKE` otherwise, detected at open time.
- **ChromaDB** (persistent client under `Settings.data_dir`) holds
  embeddings of every turn and note for recall by meaning. It is an index,
  not a record: anything in it can be rebuilt from SQLite. An
  `InMemoryVectorStore` with a hashing bag-of-words embedding implements
  the same `VectorStore` Protocol for tests and ChromaDB-free runs.
- **Facts are first-class rows** (`key`, `value`, `source`, `confidence`,
  timestamps), not free text. The user profile (name, preferences,
  projects, habits) is a view over the `profile.*` keys. Keyed facts can be
  updated, forgotten and ranked by confidence; a paragraph of prose cannot.
- `MemoryManager.recall` assembles profile, loose facts, recent turns and
  semantic hits into a character-budgeted context, so memory can never
  crowd the user's actual question out of the prompt. The end-of-
  conversation summary is a hook (`Summariser`); the LLM-backed version
  lands with the agent.

## Alternatives considered

- **One store: SQLite only, with embeddings in a BLOB column** – workable
  with `sqlite-vec`, but that is a C extension to build and ship for every
  platform, and brute-force cosine in Python stops scaling after a few
  thousand turns. ChromaDB is a `pip install` and persists to the same
  data directory.
- **One store: ChromaDB only** – no transactions, no ordering guarantees,
  no atomic "pop the due reminders". Metadata filters are not a relational
  query language.
- **A hosted vector database** – breaks the all-local promise and adds a
  credential for a feature that works fine on a laptop.
- **Facts as prose in the system prompt** – simple, but there is no way to
  correct one fact without rewriting the whole text, and no provenance.
- **An ORM** – six tables do not justify a dependency that would hide the
  SQL a reviewer wants to see.

## Consequences

- Privacy: every byte of memory is a file under `Settings.data_dir` on the
  user's machine. Nothing is sent anywhere by the memory layer; with the
  LOCAL runtime mode the whole assistant is offline. Deleting the directory
  forgets everything.
- Two stores can drift; the manager writes SQLite first and treats Chroma
  as derived, so a rebuild-from-SQLite tool is cheap to add if drift is
  ever observed.
- ChromaDB's default embedding model is downloaded on first use (about
  80 MB). The test-suite never does this: contract tests run against the
  in-memory store, and the ChromaDB integration tests inject the hashing
  embedding function.
- Context budgets are counted in characters, not tokens, because the
  tokenizer depends on the LLM provider; the agent derives the budget from
  its model's window.
