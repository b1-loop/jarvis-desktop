// Compiles the miniaudio implementation exactly once. This translation unit
// is built as its own target without our warnings-as-errors policy because
// the code is third-party; see audio_engine/CMakeLists.txt.
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>
