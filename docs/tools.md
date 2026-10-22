# Tools

Every capability Jarvis can use is a plugin in `src/jarvis/tools/`,
discovered automatically (ADR-0006). "Confirmation" is what the central
`ConfirmationPolicy` asks before the call runs; `JARVIS_CONFIRM_DESTRUCTIVE=false`
(trusted mode) switches all of it off.

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
