# Tool catalogue

Every capability Jarvis can use is a plugin in `src/jarvis/tools/`, discovered
automatically (see [ADR-0006](adr/0006-plugin-tools-and-confirmation.md) and
`CONTRIBUTING.md`, "Adding a tool"). This page lists what exists, whether the
agent asks before running it, and which service it goes through. The LLM sees
each tool's `description` and input schema; the descriptions in the code are
the authoritative wording.

Claude is offered every tool on every request. Ollama and llama.cpp get at
most `JARVIS_MAX_TOOLS_PER_TURN` (default 12) per round: the core set
(`JARVIS_CORE_TOOLS`, default `current_time` and `web_search`), the tools
already used in the turn, and the best matches for the request
([ADR-0020](adr/0020-tool-selection-for-small-models.md)).

**Confirmation** column: *yes* means the central `ConfirmationPolicy` asks the
user every time; *per call* means only some argument combinations ask (stated
in the row); *no* means the call is read-only or a reversible change the user
asked for by name. `JARVIS_CONFIRM_DESTRUCTIVE=false` (trusted mode) switches
all confirmation off.

## Computer control

| Tool | What it does | Confirmation | Service |
| --- | --- | --- | --- |
| `open_app` | Launch an app, program path or URL by name | no | `Platform.launch` |
| `close_app` | Close every window of an app gracefully; `force` kills it | yes (stronger prompt with `force`) | `Platform.close_process_by_name` |
| `list_windows` | Visible windows with title, executable and pid, front-most first | no | `Platform.list_windows` |
| `switch_window` | Focus the front-most window whose title contains the text | no | `Platform.focus_window_by_title` |
| `window_action` | Minimize, maximize, restore or close one window (focused or by title) | per call: `close` | `Platform` window methods |
| `get_volume` | Master volume and mute state | no | `Platform.get_volume`, `is_muted` |
| `set_volume` | Set volume to a level or by a step; unmutes above 0 | no | `Platform.set_volume`, `set_muted` |
| `mute` | Mute or unmute all output | no | `Platform.set_muted` |
| `set_brightness` | Set display brightness to a level or by a step; says plainly when the screen has no software control | no | `Platform.set_brightness` |
| `take_screenshot` | Save a PNG of the primary monitor, all monitors or the focused window under `<data_dir>/screenshots` and return the path | no | `Platform.screenshot`, `screenshot_window` |
| `lock_screen` | Lock the workstation | yes | `Platform.lock_screen` |
| `type_text` | Type text at the cursor in the focused window, optionally pressing Enter | yes (prompt previews the text) | `Platform.type_text` |
| `clipboard_get` | Read the clipboard text for the LLM | no | `Platform.get_clipboard_text` |
| `clipboard_set` | Put text on the clipboard | no | `Platform.set_clipboard_text` |
| `notify` | Show a desktop notification | no | `Platform.notify` |
| `set_hud_colour` | Recolour the HUD and its orb: a preset (red, green, violet, ice, pink ...), any colour word in English or Swedish, or a hex code as a custom accent; applied live and remembered ([ADR-0028](adr/0028-hud-colour-themes.md)) | no | `ToolContext.hud` (the HUD settings page) |

## Files

| Tool | What it does | Confirmation | Service |
| --- | --- | --- | --- |
| `find_files` | Find documents by name words (optionally content) under `JARVIS_DOCUMENT_ROOTS`, newest first | no | `integrations.documents.find_documents` |
| `recent_files` | Most recently changed documents; optionally open the newest ("open the last thing I worked on") | no | `integrations.documents.recent_documents`, `Platform.launch` |
| `open_file` | Open a file in its default app; programs, scripts and shortcuts are refused | no | `Platform.launch` |
| `summarize_document` | Extract a PDF/Word/Markdown/text/CSV file's text within a budget, headings kept, for the LLM to summarise | no | `integrations.documents.extract_text`, `summarise_for_llm` |

## Dev mode

All commands run through `utils/sandbox.py` with the user's policy: working
root (`JARVIS_SHELL_WORKING_ROOT`), scrubbed environment, timeout, output cap.

| Tool | What it does | Confirmation | Service |
| --- | --- | --- | --- |
| `git_status` | Branch, ahead/behind, staged, modified and untracked files | no | `run_sandboxed` (`git status --porcelain`) |
| `git_log` | Last N commits: hash, author, relative date, subject | no | `run_sandboxed` (`git log`) |
| `git_diff_summary` | Files and lines changed (unstaged, staged or all) | no | `run_sandboxed` (`git diff --stat`) |
| `git_push` | Push the current or a named branch; never forces; ref names validated | yes | `run_sandboxed` (`git push`) |
| `run_tests` | Detect cargo / npm / pytest from project files and run the suite (up to an hour) | no | `run_sandboxed` |
| `open_in_vscode` | Open a folder or file with VS Code's `code` launcher | no | `run_sandboxed` (`code`) |
| `start_dev_server` | Start a detected or named dev server in the background; returns pid and log path; its exit is announced later | yes | `spawn_sandboxed`, `BackgroundJobs` (ADR-0017) |
| `explain_stack_trace` | Parse a Python, JavaScript or C++ (gdb, WinDbg) trace into exception and innermost-first frames with a likely culprit; nothing is executed | no | pure parsing |
| `shell_command` | Run a command line under the sandbox policy | per call: unknown binaries and risky patterns | `run_sandboxed` |

## Notes, todos and reminders

| Tool | What it does | Confirmation | Service |
| --- | --- | --- | --- |
| `add_note` | Save a dictated note, optionally titled | no | `MemoryManager.add_note` |
| `search_notes` | Full-text search over notes | no | `MemoryManager.search_notes` |
| `add_todo` | Add a todo item | no | `MemoryManager.add_todo` |
| `list_todos` | Open (or all) todos with ids | no | `MemoryManager.list_todos` |
| `complete_todo` | Tick off a todo by id or by an unambiguous text fragment | no | `MemoryManager.complete_todo` |
| `add_reminder` | Reminder `in_minutes` from now (resolved against the clock) or at an ISO `due_at`; exactly one, naive means local, past and far-future times rejected with the current time in the error | no | `MemoryManager.add_reminder` |
| `list_reminders` | Pending (or all) reminders in local time | no | `MemoryManager.list_reminders` |

## System and information

| Tool | What it does | Confirmation | Service |
| --- | --- | --- | --- |
| `system_monitor` | Whole-computer CPU, RAM, disk and battery, plus the heaviest processes, labelled apart (about 0.2 s) | no | psutil totals + one-call process table (ADR-0026) |
| `battery_status` | Battery level, charging state and time remaining | no | psutil probe |
| `current_time` | Current date, time and zone, optionally converted | no | `Clock` |

## Screen understanding

Captures happen only inside one of these calls; each says aloud when the
screenshot went to a cloud provider ([ADR-0016](adr/0016-screen-understanding.md)).
They need a vision-capable model: Claude, or Ollama with
`JARVIS_OLLAMA_VISION_MODEL`; with llama.cpp they say they are unavailable.

| Tool | What it does | Confirmation | Service |
| --- | --- | --- | --- |
| `describe_screen` | Describe what is on the screen, optionally one aspect of it | no | `ToolContext.vision` (`ScreenReader`) |
| `explain_screen_error` | Read the error on screen and explain it with a likely fix | no | `ToolContext.vision` |
| `ask_about_screen` | Answer the user's own question about the screen | no | `ToolContext.vision` |

## Why these confirmation choices

- Confirmation is for actions that are hard to undo, act on something other
  than what the user named, publish work or run code indefinitely: killing
  or closing apps (unsaved work), locking, typing into whatever has focus,
  pushing, starting a server.
- Volume, brightness, clipboard, opening apps and files, and writing to
  Jarvis's own notes, todos and reminders are exactly what the user asked
  for and are undone by the next sentence. Asking first would make the
  assistant unusable without adding safety.
- Tools that need a service the session lacks (no OS backend in text-only
  mode, memory disabled) fail with a clear "not available in this session"
  result instead of pretending.

## Integration tools

External services reach these tools only through `ToolContext.integrations`
(ADR-0018). Without credentials each service uses its `Stub*` provider
(ADR-0009), and when no integrations are wired at all every tool answers
"that integration isn't configured" with the variables to set.

| Tool | What it does | Confirmation | Provider |
| --- | --- | --- | --- |
| `list_unread_email` | Unread mail across mailboxes, newest first, with `provider:id` references | No | Gmail, Outlook, `StubEmailProvider` |
| `summarize_inbox` | Unread mail with trimmed bodies and per-sender counts, for the LLM to summarise | No | Gmail, Outlook, stub |
| `search_email` | Free-text search over all mail; top hits include bodies to answer from | No | Gmail, Outlook, stub |
| `draft_email_reply` | Creates a reply draft in the mailbox and reads it back; sends nothing | No (the send is confirmed) | Gmail, Outlook, stub |
| `send_email` | Sends a draft from this session; refuses if the restated recipients/subject/first line differ | **Always** – reads recipient, subject, first line | Gmail, Outlook, stub |
| `todays_agenda` | Events of today (or a day up to two weeks ahead) across calendars | No | Google Calendar, Outlook, `StubCalendarProvider` |
| `upcoming_events` | Events starting in the next N minutes; same logic feeds the meeting trigger | No | Google Calendar, Outlook, stub |
| `find_free_slot` | Free slots of a given length in working hours, free on every calendar | No | Google Calendar, Outlook, stub |
| `create_event` | Creates an event and invites attendees | **Always** – reads title, day/time, length, attendees | Google Calendar, Outlook, stub |
| `web_search` | Ranked links and snippets | No | DuckDuckGo (keyless), Brave (`BRAVE_SEARCH_API_KEY`) |
| `read_article` | Readable text of a web page; refuses local/private addresses | No | `fetch_article` over the shared HTTP client |
| `summarize_youtube` | Video transcript, sampled evenly across the video to fit a budget | No | `youtube-transcript-api` |
| `read_messages` | Unread, mentioning or recent chat messages across providers | No | Slack, Discord, Teams, `StubChatProvider` |
| `send_message` | Posts to a channel or chat | **Always** – reads channel and text (asks which service first if several and none named) | Slack, Discord, Teams, stub |
| `music_control` | Pause, resume/play, next | No – own device, free, instantly reversible | Spotify, `StubMusicProvider` |
| `play_music` | Searches and plays the best match | No | Spotify, stub |
| `queue_music` | Adds the best match to the queue | No | Spotify, stub |
| `now_playing` | Current track, artist and position | No | Spotify, stub |
| `list_devices` | Smart-home devices with id, kind, state, brightness | No | Home Assistant (incl. Hue), `StubSmartHomeProvider` |
| `set_device` | On/off, brightness, scene | **Per device** – only `lock`, `alarm_control_panel`, `climate`, `water_heater` domains | Home Assistant, stub |
| `draft_document` | Writes LLM-composed Markdown to `.md` or `.docx` in `JARVIS_DOCUMENTS_OUTPUT_DIR`, optionally citing an email | **Only when overwriting** an existing file | Local files (python-docx); Google Docs out of scope |

Notes:

- Ids in tool results (`gmail:18c2…`, `stub-email:stub-draft-1`) carry the
  provider that issued them; pass them back unchanged.
- With several providers of one kind, listings are merged and a failing
  provider is reported in `detail`/`data.failures` while the others' results
  are still returned.
- Google Docs output would need a Drive/Docs OAuth scope on the shared
  Google sign-in; it is left out so Gmail users are not asked for it. A
  `.docx` uploads to Google Docs losslessly.
