# pybind11 modules. One module per native component; each is a thin layer
# that converts between Python and C++ types and releases the GIL around
# blocking calls. All logic lives in the component libraries, never here.

# Defines a Jarvis extension module and places the built file directly in the
# Python package so no copy step is needed. The $<0:> generator expression
# stops multi-config generators from appending a per-configuration subfolder.
function(jarvis_add_python_module name)
    pybind11_add_module(${name} MODULE ${ARGN})
    target_link_libraries(${name} PRIVATE jarvis::common)
    jarvis_set_warnings(${name})
    set_target_properties(${name} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${JARVIS_PYTHON_NATIVE_DIR}/$<0:>"
        RUNTIME_OUTPUT_DIRECTORY "${JARVIS_PYTHON_NATIVE_DIR}/$<0:>")
endfunction()

jarvis_add_python_module(jarvis_audio jarvis_audio_module.cpp)
target_link_libraries(jarvis_audio PRIVATE jarvis::audio_engine)
jarvis_copy_onnxruntime_dll(jarvis_audio)

jarvis_add_python_module(jarvis_os jarvis_os_module.cpp)
jarvis_add_python_module(jarvis_inference jarvis_inference_module.cpp)
jarvis_add_python_module(jarvis_overlay jarvis_overlay_module.cpp)
