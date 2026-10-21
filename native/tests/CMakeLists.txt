# Catch2 unit tests for every native component. One executable, discovered
# test by test by ctest so failures are reported individually.

include(FetchContent)
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE
    SYSTEM)
FetchContent_MakeAvailable(Catch2)

add_executable(jarvis_tests
    version_test.cpp)

target_link_libraries(jarvis_tests PRIVATE
    jarvis::common
    Catch2::Catch2WithMain)
jarvis_set_warnings(jarvis_tests)

list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
include(Catch)
catch_discover_tests(jarvis_tests)
