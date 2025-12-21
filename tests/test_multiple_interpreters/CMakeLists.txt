# CMakeLists.txt -- Build system for the pybind11 multiple interpreters test suite
#
# Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

set(PYBIND11_MULTIPLE_INTERPRETERS_TEST_FILES test_multiple_interpreters.py)

# These modules don't get mixed with other test modules because those aren't subinterpreter safe.
pybind11_add_module(mod_per_interpreter_gil THIN_LTO mod_per_interpreter_gil.cpp)
pybind11_add_module(mod_shared_interpreter_gil THIN_LTO mod_shared_interpreter_gil.cpp)
pybind11_enable_warnings(mod_per_interpreter_gil)
pybind11_enable_warnings(mod_shared_interpreter_gil)

# Put the built modules next to `pybind11_tests.so` so that the test scripts can find them.
get_target_property(pybind11_tests_output_directory pybind11_tests LIBRARY_OUTPUT_DIRECTORY)
set_target_properties(mod_per_interpreter_gil PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                         "${pybind11_tests_output_directory}")
set_target_properties(mod_shared_interpreter_gil PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                            "${pybind11_tests_output_directory}")

if(PYBIND11_TEST_SMART_HOLDER)
  target_compile_definitions(
    mod_per_interpreter_gil
    PUBLIC -DPYBIND11_RUN_TESTING_WITH_SMART_HOLDER_AS_DEFAULT_BUT_NEVER_USE_IN_PRODUCTION_PLEASE)
  target_compile_definitions(
    mod_shared_interpreter_gil
    PUBLIC -DPYBIND11_RUN_TESTING_WITH_SMART_HOLDER_AS_DEFAULT_BUT_NEVER_USE_IN_PRODUCTION_PLEASE)
endif()

add_dependencies(pytest mod_per_interpreter_gil mod_shared_interpreter_gil)

# Convert relative to full file names and add to pytest test files
list(TRANSFORM PYBIND11_MULTIPLE_INTERPRETERS_TEST_FILES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")
list(APPEND PYBIND11_ABS_PYTEST_FILES ${PYBIND11_MULTIPLE_INTERPRETERS_TEST_FILES})
set(PYBIND11_ABS_PYTEST_FILES
    ${PYBIND11_ABS_PYTEST_FILES}
    PARENT_SCOPE)
