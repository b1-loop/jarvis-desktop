# Tool catalogue

Every capability Jarvis can use is a plugin in `src/jarvis/tools/`, discovered
automatically (see [ADR-0006](adr/0006-plugin-tools-and-confirmation.md) and
`CONTRIBUTING.md`, "Adding a tool"). This page lists what exists, whether the
agent asks before running it, and which service it goes through. The LLM sees
each tool's `description` and input schema; the descriptions in the code are
the authoritative wording.

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
| `add_reminder` | Reminder at an ISO datetime (from the LLM); naive means local, past times rejected | no | `MemoryManager.add_reminder` |
| `list_reminders` | Pending (or all) reminders in local time | no | `MemoryManager.list_reminders` |

## System and information

| Tool | What it does | Confirmation | Service |
| --- | --- | --- | --- |
| `system_monitor` | CPU, memory, disk, battery and the heaviest processes | no | psutil probe |
| `battery_status` | Battery level, charging state and time remaining | no | psutil probe |
| `current_time` | Current date, time and zone, optionally converted | no | `Clock` |

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
