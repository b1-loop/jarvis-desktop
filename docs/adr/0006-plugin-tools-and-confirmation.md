# 0006. Plugin tools, central confirmation and a policy-first sandbox

Date: 2026-10-21
Status: Accepted

## Context

Jarvis will grow to several dozen capabilities (shell, files, email,
calendar, music, smart home, ...) written over many steps, and the LLM
chooses among them at run time. Some of them change the world: delete,
send, spend, control devices. Four questions had to be answered once, before
the first tool was written:

- How does a new capability get into the system?
- Who makes sure a dangerous call is confirmed by the user?
- How does a tool reach the OS, memory and settings without coupling the
  tool layer to concrete providers?
- How are shell commands, the most powerful and most abusable tool, kept
  safe when the command text comes from an LLM?

## Decision

**Plugins are discovered, never registered.** A tool is one module in
`jarvis.tools` that defines a concrete `Tool` subclass with `name`,
`description`, a pydantic `Input` model and `async run()`.
`ToolRegistry.discover()` walks the package with `pkgutil`, imports each
public module and instantiates every tool it finds. There is no list to
edit; the module's existence is the registration. A module that fails to
import, declares an invalid tool or reuses a name is logged, recorded as a
`DiscoveryFailure` for the dashboard and skipped, so one broken plugin
cannot stop the assistant from starting.

**Confirmation is enforced centrally.** Tools only *declare* risk:
`requires_confirmation = True` for tools that always change something, or an
overridden `needs_confirmation(params, context)` when the risk depends on
the arguments (the shell tool inspects the command). `ConfirmationPolicy`
is the single place that turns those declarations, a configurable list of
always-confirm categories and the `JARVIS_CONFIRM_DESTRUCTIVE` trusted-mode
flag into a decision, asks the user through `context.ask_confirmation`,
publishes `ConfirmationRequested`/`ConfirmationResolved` events and returns
a plain `ok=False` result when the user declines. A tool never asks the
user itself and can never forget to.

**Services arrive through `ToolContext`, typed as Protocols.** The context
carries settings, the event bus, a `Clock`, the confirmation callback and
placeholders for the OS facade and memory. Tools depend on those shapes,
never on `jarvis.os_integration.windows` or a memory backend, so every tool
is unit-tested with fakes and the composition root is the only module that
knows which implementation is wired in.

**The sandbox is policy first, process second.** `utils/sandbox.py` parses
the command into an argv (no shell is ever spawned), applies a denylist of
destructive patterns (refused outright) and risky ones (confirmation), an
allowlist of binaries from settings, a working-directory root, an
environment allowlist that withholds every credential, a timeout that kills
the process and an output cap. `check_command()` produces the verdict the
shell tool uses to decide on confirmation, so the user is asked about
exactly what will run.

## Alternatives considered

- **Explicit registration list or entry points** – a list is one more file
  to touch and merge-conflict on for every tool; entry points need a
  reinstall and hide the plugin set in packaging metadata. Both make the
  "drop in a file" contract false.
- **Each tool asks for confirmation itself** – the simplest thing to write
  and the easiest to forget. A review cannot prove that every future tool
  asks; a central policy can be tested once.
- **Tools import the services they need** – shorter code, but the tool
  layer would then import Win32 wrappers and database clients, which breaks
  text-only mode on machines without the native build and makes tests need
  the real thing.
- **Run shell commands through `cmd /c` or `bash -c`** – gives the LLM
  pipes and redirects for free, and also gives it `&& rm -rf`, variable
  expansion and every other escape hatch. Refusing shells and operators
  costs some convenience and buys a policy that can actually be reasoned
  about.
- **Denylist only, no allowlist** – a denylist can only forbid what someone
  thought of. The allowlist makes the default posture "ask" for anything
  unknown, with the denylist as a hard floor beneath it.

## Consequences

- Adding a capability is one file plus one test file; CONTRIBUTING.md's
  "Adding a tool" section is the whole procedure.
- Risk declarations live next to the code that creates the risk, while the
  decision and the user interaction live in one tested module.
- Trusted mode exists for power users but is a settings flag only; nothing
  spoken at run time can switch confirmation off.
- The sandbox refuses some legitimate commands (`cmd /c dir`, pipes). The
  remedy is a dedicated tool for the task, not a looser policy.
- The OS and memory Protocols in `ToolContext` are placeholders; when the
  real interfaces land (steps 4 and 9) the field types change but no tool
  code does, because the tools only ever used the Protocol surface.
