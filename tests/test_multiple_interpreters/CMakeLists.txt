# CMakeLists.txt -- Build system for the pybind11 multiple interpreters test suite
#
# Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

set(PYBIND11_MULTIPLE_INTERPRETERS_TEST_FILES test_multiple_interpreters.py)

set(modules mod_per_interpreter_gil mod_shared_interpreter_gil
            mod_per_interpreter_gil_with_singleton)

# These modules don't get mixed with other test modules because those aren't subinterpreter safe.
foreach(mod IN LISTS modules)
  pybind11_add_module("${mod}" THIN_LTO "${mod}.cpp")
  pybind11_enable_warnings("${mod}")
endforeach()

# Put the built modules next to `pybind11_tests.so` so that the test scripts can find them.
get_target_property(pybind11_tests_output_directory pybind11_tests LIBRARY_OUTPUT_DIRECTORY)
foreach(mod IN LISTS modules)
  set_target_properties("${mod}" PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                            "${pybind11_tests_output_directory}")
endforeach()

if(PYBIND11_TEST_SMART_HOLDER)
  foreach(mod IN LISTS modules)
    target_compile_definitions(
      "${mod}"
      PUBLIC -DPYBIND11_RUN_TESTING_WITH_SMART_HOLDER_AS_DEFAULT_BUT_NEVER_USE_IN_PRODUCTION_PLEASE
    )
  endforeach()
endif()

add_dependencies(pytest ${modules})

# Convert relative to full file names and add to pytest test files
list(TRANSFORM PYBIND11_MULTIPLE_INTERPRETERS_TEST_FILES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND PYBIND11_ABS_PYTEST_FILES ${PYBIND11_MULTIPLE_INTERPRETERS_TEST_FILES})
set(PYBIND11_ABS_PYTEST_FILES
    ${PYBIND11_ABS_PYTEST_FILES}
    PARENT_SCOPE)
