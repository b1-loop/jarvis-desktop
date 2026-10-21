# third_party

Git submodules for whisper.cpp and llama.cpp are added here in step 12
(LOCAL mode). Nothing in this directory is linted or formatted by our tools;
`.clang-format` and `.clang-tidy` exclude it and CI checks out submodules
recursively only for the native lane.
