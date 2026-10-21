# Applies the project-wide warning policy to a target.
#
# Warnings are errors on every compiler so that CI on Windows (MSVC) and a
# future Linux lane (GCC/Clang) enforce the same bar. Third-party code is
# consumed through SYSTEM includes and is therefore exempt.

function(jarvis_set_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            /W4
            /WX
            /permissive-
            /utf-8
            /EHsc
            /Zc:__cplusplus
            /external:anglebrackets
            /external:W0
            # C4324 reports padding caused by alignas; padding is the whole
            # point of aligning atomics to separate cache lines.
            /wd4324)
    else()
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Werror
            -Wshadow
            -Wconversion
            -Wsign-conversion
            -Wnon-virtual-dtor
            -Wold-style-cast)
    endif()
endfunction()
