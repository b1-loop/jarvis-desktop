# Architecture Decision Records

Short, dated records of the decisions that shape Jarvis and the reasons behind
them. A reviewer who asks "why did you do it this way?" should find the answer
here. New ADRs follow `template.md`; superseded ones are kept and marked.

| # | Title | Status |
| --- | --- | --- |
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-decoupled-native-build.md) | Build native extensions separately from the Python package | Accepted |
| [0003](0003-cpp-audio-hot-path.md) | C++ for the audio hot path | Accepted |
| [0004](0004-win32-behind-platform-interface.md) | Win32 behind a Platform interface | Accepted |
| [0005](0005-provider-interfaces-and-streaming.md) | Provider interfaces, streaming speech and a third LLM provider | Accepted |
| [0006](0006-plugin-tools-and-confirmation.md) | Plugin tools, central confirmation and a policy-first sandbox | Accepted |
| [0007](0007-proactive-module.md) | Proactive module: one scheduler, polled triggers, YAML routines | Accepted |
| [0008](0008-memory-storage.md) | SQLite for structured memory, ChromaDB for semantic memory | Accepted |
| [0009](0009-integration-providers.md) | Integration providers: interfaces, stub by default, least privilege | Accepted |
| [0010](0010-local-inference-wrappers.md) | Own thin C++ wrappers over whisper.cpp and llama.cpp | Accepted |
| [0011](0011-imgui-overlay.md) | Dear ImGui on DirectX 11 for the status overlay | Accepted |
| [0012](0012-web-and-document-access.md) | Web search, article reading and local document access | Accepted |
| [0013](0013-cloud-vs-local-modes.md) | CLOUD vs LOCAL runtime modes | Accepted |
| [0014](0014-agent-loop-and-orchestrator.md) | Agent loop, task chaining and one orchestrator for text and voice | Accepted |
| [0015](0015-tray-and-dashboard.md) | Tray icon with pystray, dashboard window with tkinter | Accepted |
| [0016](0016-screen-understanding.md) | Screen understanding: downscale locally, byte budget, ask only on request | Accepted |
| [0017](0017-background-processes-for-dev-tools.md) | Background processes for dev tools: spawn under the sandbox, track in memory | Accepted |
| [0018](0018-integration-services-bundle.md) | One IntegrationServices bundle on ToolContext | Accepted |
| [0019](0019-voice-pipeline-and-barge-in.md) | Voice pipeline: sentence pipelining, native-first barge-in, honest degradation | Accepted |
| [0020](0020-tool-selection-for-small-models.md) | Per-turn tool selection and a grounding guard for small local models | Accepted |
| [0021](0021-desktop-launcher.md) | Desktop launcher: uvw and shortcuts, not a frozen executable (yet) | Accepted |
| [0022](0022-overlay-animations-and-transparency.md) | Overlay animations and per-pixel transparency | Accepted |
| [0023](0023-overlay-gpu-bloom-and-scene.md) | Overlay GPU bloom, procedural 3D scene and a showpiece corner HUD | Accepted |
| [0024](0024-interactive-hud.md) | Interactive HUD: hit-tested shapes, explicit focus, polled UI events, user settings | Accepted |
| [0025](0025-relaunch-replays-the-hud.md) | A launch while running replays the HUD; a wake brings a closed HUD back | Accepted |
| [0026](0026-one-call-process-table.md) | system_monitor reads the process table in one kernel call | Accepted |
| [0027](0027-capture-watchdog.md) | Capture recovery: native detection, a Python watchdog | Accepted |
