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
| 0013 | CLOUD vs LOCAL runtime modes *(planned, step 12)* | Proposed |

