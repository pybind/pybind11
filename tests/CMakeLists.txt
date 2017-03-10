# CMakeLists.txt -- Build system for the pybind11 test suite
#
# Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 2.8.12)

option(PYBIND11_WERROR  "Report all warnings as errors"  OFF)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    # We're being loaded directly, i.e. not via add_subdirectory, so make this
    # work as its own project and load the pybind11Config to get the tools we need
    project(pybind11_tests)

    find_package(pybind11 REQUIRED CONFIG)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting tests build type to MinSizeRel as none was specified")
  set(CMAKE_BUILD_TYPE MinSizeRel CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# Full set of test files (you can override these; see below)
set(PYBIND11_TEST_FILES
  test_alias_initialization.cpp
  test_buffers.cpp
  test_callbacks.cpp
  test_chrono.cpp
  test_class_args.cpp
  test_constants_and_functions.cpp
  test_copy_move_policies.cpp
  test_docstring_options.cpp
  test_eigen.cpp
  test_enum.cpp
  test_eval.cpp
  test_exceptions.cpp
  test_inheritance.cpp
  test_issues.cpp
  test_keep_alive.cpp
  test_kwargs_and_defaults.cpp
  test_methods_and_attributes.cpp
  test_modules.cpp
  test_multiple_inheritance.cpp
  test_numpy_array.cpp
  test_numpy_dtypes.cpp
  test_numpy_vectorize.cpp
  test_opaque_types.cpp
  test_operator_overloading.cpp
  test_pickling.cpp
  test_python_types.cpp
  test_sequences_and_iterators.cpp
  test_smart_ptr.cpp
  test_stl_binders.cpp
  test_virtual_functions.cpp
)

# Invoking cmake with something like:
#     cmake -DPYBIND11_TEST_OVERRIDE="test_issues.cpp;test_picking.cpp" ..
# lets you override the tests that get compiled and run.  You can restore to all tests with:
#     cmake -DPYBIND11_TEST_OVERRIDE= ..
if (PYBIND11_TEST_OVERRIDE)
  set(PYBIND11_TEST_FILES ${PYBIND11_TEST_OVERRIDE})
endif()

string(REPLACE ".cpp" ".py" PYBIND11_PYTEST_FILES "${PYBIND11_TEST_FILES}")

# Check if Eigen is available; if not, remove from PYBIND11_TEST_FILES (but
# keep it in PYBIND11_PYTEST_FILES, so that we get the "eigen is not installed"
# skip message).
list(FIND PYBIND11_TEST_FILES test_eigen.cpp PYBIND11_TEST_FILES_EIGEN_I)
if(PYBIND11_TEST_FILES_EIGEN_I GREATER -1)
  # Try loading via newer Eigen's Eigen3Config first (bypassing tools/FindEigen3.cmake).
  # Eigen 3.3.1+ exports a cmake 3.0+ target for handling dependency requirements, but also
  # produces a fatal error if loaded from a pre-3.0 cmake.
  if (NOT CMAKE_VERSION VERSION_LESS 3.0)
    find_package(Eigen3 QUIET CONFIG)
    if (EIGEN3_FOUND)
      if (EIGEN3_VERSION_STRING AND NOT EIGEN3_VERSION_STRING VERSION_LESS 3.3.1)
        set(PYBIND11_EIGEN_VIA_TARGET 1)
      endif()
    endif()
  endif()
  if (NOT EIGEN3_FOUND)
    # Couldn't load via target, so fall back to allowing module mode finding, which will pick up
    # tools/FindEigen3.cmake
    find_package(Eigen3 QUIET)
  endif()

  if(EIGEN3_FOUND)
    # Eigen 3.3.1+ cmake sets EIGEN3_VERSION_STRING (and hard codes the version when installed
    # rather than looking it up in the cmake script); older versions, and the
    # tools/FindEigen3.cmake, set EIGEN3_VERSION instead.
    if(NOT EIGEN3_VERSION AND EIGEN3_VERSION_STRING)
      set(EIGEN3_VERSION ${EIGEN3_VERSION_STRING})
    endif()
    message(STATUS "Building tests with Eigen v${EIGEN3_VERSION}")
  else()
    list(REMOVE_AT PYBIND11_TEST_FILES ${PYBIND11_TEST_FILES_EIGEN_I})
    message(STATUS "Building tests WITHOUT Eigen")
  endif()
endif()

# Compile with compiler warnings turned on
function(pybind11_enable_warnings target_name)
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4)
  else()
      target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wconversion -Wcast-qual)
  endif()

  if(PYBIND11_WERROR)
    if(MSVC)
      target_compile_options(${target_name} PRIVATE /WX)
    else()
      target_compile_options(${target_name} PRIVATE -Werror)
    endif()
  endif()
endfunction()


# Create the binding library
pybind11_add_module(pybind11_tests THIN_LTO pybind11_tests.cpp
  ${PYBIND11_TEST_FILES} ${PYBIND11_HEADERS})

pybind11_enable_warnings(pybind11_tests)

if(EIGEN3_FOUND)
  if (PYBIND11_EIGEN_VIA_TARGET)
    target_link_libraries(pybind11_tests PRIVATE Eigen3::Eigen)
  else()
    target_include_directories(pybind11_tests PRIVATE ${EIGEN3_INCLUDE_DIR})
  endif()
  target_compile_definitions(pybind11_tests PRIVATE -DPYBIND11_TEST_EIGEN)
endif()

set(testdir ${CMAKE_CURRENT_SOURCE_DIR})

# Always write the output file directly into the 'tests' directory (even on MSVC)
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(pybind11_tests PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${testdir})
  foreach(config ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${config} config)
    set_target_properties(pybind11_tests PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${config} ${testdir})
  endforeach()
endif()

# Make sure pytest is found or produce a fatal error
if(NOT PYBIND11_PYTEST_FOUND)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import pytest; print(pytest.__version__)"
                  RESULT_VARIABLE pytest_not_found OUTPUT_VARIABLE pytest_version ERROR_QUIET)
  if(pytest_not_found)
    message(FATAL_ERROR "Running the tests requires pytest. Please install it manually"
                        " (try: ${PYTHON_EXECUTABLE} -m pip install pytest)")
  elseif(pytest_version VERSION_LESS 3.0)
    message(FATAL_ERROR "Running the tests requires pytest >= 3.0. Found: ${pytest_version}"
                        "Please update it (try: ${PYTHON_EXECUTABLE} -m pip install -U pytest)")
  endif()
  set(PYBIND11_PYTEST_FOUND TRUE CACHE INTERNAL "")
endif()

# A single command to compile and run the tests
add_custom_target(pytest COMMAND ${PYTHON_EXECUTABLE} -m pytest ${PYBIND11_PYTEST_FILES}
                  DEPENDS pybind11_tests WORKING_DIRECTORY ${testdir})

if(PYBIND11_TEST_OVERRIDE)
  add_custom_command(TARGET pytest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Note: not all tests run: -DPYBIND11_TEST_OVERRIDE is in effect")
endif()

# Add a check target to run all the tests, starting with pytest (we add dependencies to this below)
add_custom_target(check DEPENDS pytest)

# The remaining tests only apply when being built as part of the pybind11 project, but not if the
# tests are being built independently.
if (NOT PROJECT_NAME STREQUAL "pybind11")
  return()
endif()

# Add a post-build comment to show the .so size and, if a previous size, compare it:
add_custom_command(TARGET pybind11_tests POST_BUILD
  COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/libsize.py
  $<TARGET_FILE:pybind11_tests> ${CMAKE_CURRENT_BINARY_DIR}/sosize-$<TARGET_FILE_NAME:pybind11_tests>.txt)

# Test CMake build using functions and targets from subdirectory or installed location
add_custom_target(test_cmake_build)
if(NOT CMAKE_VERSION VERSION_LESS 3.1)
  # 3.0 needed for interface library for subdirectory_target/installed_target
  # 3.1 needed for cmake -E env for testing

  include(CMakeParseArguments)
  function(pybind11_add_build_test name)
    cmake_parse_arguments(ARG "INSTALL" "" "" ${ARGN})

    set(build_options "-DCMAKE_PREFIX_PATH=${PROJECT_BINARY_DIR}/mock_install"
                      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                      "-DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
                      "-DPYBIND11_CPP_STANDARD=${PYBIND11_CPP_STANDARD}")
    if(NOT ARG_INSTALL)
      list(APPEND build_options "-DPYBIND11_PROJECT_DIR=${PROJECT_SOURCE_DIR}")
    endif()

    add_custom_target(test_${name} ${CMAKE_CTEST_COMMAND}
      --quiet --output-log test_cmake_build/${name}.log
      --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/test_cmake_build/${name}"
                       "${CMAKE_CURRENT_BINARY_DIR}/test_cmake_build/${name}"
      --build-config Release
      --build-noclean
      --build-generator ${CMAKE_GENERATOR}
      $<$<BOOL:${CMAKE_GENERATOR_PLATFORM}>:--build-generator-platform> ${CMAKE_GENERATOR_PLATFORM}
      --build-makeprogram ${CMAKE_MAKE_PROGRAM}
      --build-target check
      --build-options ${build_options}
    )
    if(ARG_INSTALL)
      add_dependencies(test_${name} mock_install)
    endif()
    add_dependencies(test_cmake_build test_${name})
  endfunction()

  pybind11_add_build_test(subdirectory_function)
  pybind11_add_build_test(subdirectory_target)

  if(PYBIND11_INSTALL)
    add_custom_target(mock_install ${CMAKE_COMMAND}
      "-DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/mock_install"
      -P "${PROJECT_BINARY_DIR}/cmake_install.cmake"
    )

    pybind11_add_build_test(installed_function INSTALL)
    pybind11_add_build_test(installed_target INSTALL)
  endif()
endif()

add_dependencies(check test_cmake_build)
