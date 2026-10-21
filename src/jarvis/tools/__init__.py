"""Tool plugins, one file per capability, discovered automatically.

The plugin contract (enforced by ``jarvis.agent.registry.ToolRegistry``):

1. Put one public module in this package (``tools/<name>.py``). Modules
   whose name starts with ``_`` are ignored.
2. Define a concrete subclass of ``jarvis.agent.tool.Tool`` in it with the
   class attributes ``name`` (snake_case, unique across all plugins),
   ``description`` (what the LLM reads to decide when to call it), ``Input``
   (a pydantic model, ideally deriving from ``ToolInput``), optionally
   ``category`` and ``requires_confirmation``, and an ``async run()``.
3. The constructor must work with no arguments; services arrive through the
   ``ToolContext`` passed to ``run``, never through imports of concrete
   providers. Optional constructor parameters (with defaults) are fine and
   are how tests inject fakes.
4. Anything that changes the system, deletes, sends, spends money or touches
   an external account sets ``requires_confirmation = True`` (or overrides
   ``needs_confirmation`` when the risk depends on the arguments). The
   confirmation policy enforces it centrally; the tool never asks itself.
5. Return a ``ToolResult``; raise ``JarvisError`` subclasses for expected
   failures. Validation errors never need handling: ``Input`` does that.
6. Add ``tests/tools/test_<name>.py`` using fakes for every service.

There is no registration list to edit. Tools must never import each other.
"""
