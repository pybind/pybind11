# CMakeLists.txt -- Build system for the pybind11 test suite
#
# Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 3.4)

# The `cmake_minimum_required(VERSION 3.4...3.18)` syntax does not work with
# some versions of VS that have a patched CMake 3.11. This forces us to emulate
# the behavior using the following workaround:
if(${CMAKE_VERSION} VERSION_LESS 3.21)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.21)
endif()

# Only needed for CMake < 3.5 support
include(CMakeParseArguments)

# Filter out items; print an optional message if any items filtered
#
# Usage:
#   pybind11_filter_tests(LISTNAME file1.cpp file2.cpp ... MESSAGE "")
#
macro(pybind11_filter_tests LISTNAME)
  cmake_parse_arguments(ARG "" "MESSAGE" "" ${ARGN})
  set(PYBIND11_FILTER_TESTS_FOUND OFF)
  foreach(filename IN LISTS ARG_UNPARSED_ARGUMENTS)
    list(FIND ${LISTNAME} ${filename} _FILE_FOUND)
    if(_FILE_FOUND GREATER -1)
      list(REMOVE_AT ${LISTNAME} ${_FILE_FOUND})
      set(PYBIND11_FILTER_TESTS_FOUND ON)
    endif()
  endforeach()
  if(PYBIND11_FILTER_TESTS_FOUND AND ARG_MESSAGE)
    message(STATUS "${ARG_MESSAGE}")
  endif()
endmacro()

macro(possibly_uninitialized)
  foreach(VARNAME ${ARGN})
    if(NOT DEFINED "${VARNAME}")
      set("${VARNAME}" "")
    endif()
  endforeach()
endmacro()

# New Python support
if(DEFINED Python_EXECUTABLE)
  set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
  set(PYTHON_VERSION "${Python_VERSION}")
endif()

# There's no harm in including a project in a project
project(pybind11_tests CXX)

# Access FindCatch and more
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../tools")

option(PYBIND11_WERROR "Report all warnings as errors" OFF)
option(DOWNLOAD_EIGEN "Download EIGEN (requires CMake 3.11+)" OFF)
option(PYBIND11_CUDA_TESTS "Enable building CUDA tests (requires CMake 3.12+)" OFF)
set(PYBIND11_TEST_OVERRIDE
    ""
    CACHE STRING "Tests from ;-separated list of *.cpp files will be built instead of all tests")
set(PYBIND11_TEST_FILTER
    ""
    CACHE STRING "Tests from ;-separated list of *.cpp files will be removed from all tests")

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  # We're being loaded directly, i.e. not via add_subdirectory, so make this
  # work as its own project and load the pybind11Config to get the tools we need
  find_package(pybind11 REQUIRED CONFIG)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT DEFINED CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting tests build type to MinSizeRel as none was specified")
  set(CMAKE_BUILD_TYPE
      MinSizeRel
      CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                               "RelWithDebInfo")
endif()

if(PYBIND11_CUDA_TESTS)
  enable_language(CUDA)
  if(DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  endif()
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Full set of test files (you can override these; see below)
set(PYBIND11_TEST_FILES
    test_async.cpp
    test_buffers.cpp
    test_builtin_casters.cpp
    test_call_policies.cpp
    test_callbacks.cpp
    test_chrono.cpp
    test_class.cpp
    test_constants_and_functions.cpp
    test_copy_move.cpp
    test_custom_type_casters.cpp
    test_docstring_options.cpp
    test_eigen.cpp
    test_enum.cpp
    test_eval.cpp
    test_exceptions.cpp
    test_factory_constructors.cpp
    test_gil_scoped.cpp
    test_iostream.cpp
    test_kwargs_and_defaults.cpp
    test_local_bindings.cpp
    test_methods_and_attributes.cpp
    test_modules.cpp
    test_multiple_inheritance.cpp
    test_numpy_array.cpp
    test_numpy_dtypes.cpp
    test_numpy_vectorize.cpp
    test_opaque_types.cpp
    test_operator_overloading.cpp
    test_pickling.cpp
    test_pytypes.cpp
    test_sequences_and_iterators.cpp
    test_smart_ptr.cpp
    test_stl.cpp
    test_stl_binders.cpp
    test_tagbased_polymorphic.cpp
    test_thread.cpp
    test_union.cpp
    test_virtual_functions.cpp)

# Invoking cmake with something like:
#     cmake -DPYBIND11_TEST_OVERRIDE="test_callbacks.cpp;test_pickling.cpp" ..
# lets you override the tests that get compiled and run.  You can restore to all tests with:
#     cmake -DPYBIND11_TEST_OVERRIDE= ..
if(PYBIND11_TEST_OVERRIDE)
  set(PYBIND11_TEST_FILES ${PYBIND11_TEST_OVERRIDE})
endif()

# You can also filter tests:
if(PYBIND11_TEST_FILTER)
  pybind11_filter_tests(PYBIND11_TEST_FILES ${PYBIND11_TEST_FILTER})
endif()

if(PYTHON_VERSION VERSION_LESS 3.5)
  pybind11_filter_tests(PYBIND11_TEST_FILES test_async.cpp MESSAGE
                        "Skipping test_async on Python 2")
endif()

# Skip tests for CUDA check:
# /pybind11/tests/test_constants_and_functions.cpp(125):
#   error: incompatible exception specifications
if(PYBIND11_CUDA_TESTS)
  pybind11_filter_tests(
    PYBIND11_TEST_FILES test_constants_and_functions.cpp MESSAGE
    "Skipping test_constants_and_functions due to incompatible exception specifications")
endif()

string(REPLACE ".cpp" ".py" PYBIND11_PYTEST_FILES "${PYBIND11_TEST_FILES}")

# Contains the set of test files that require pybind11_cross_module_tests to be
# built; if none of these are built (i.e. because TEST_OVERRIDE is used and
# doesn't include them) the second module doesn't get built.
set(PYBIND11_CROSS_MODULE_TESTS test_exceptions.py test_local_bindings.py test_stl.py
                                test_stl_binders.py)

set(PYBIND11_CROSS_MODULE_GIL_TESTS test_gil_scoped.py)

# Check if Eigen is available; if not, remove from PYBIND11_TEST_FILES (but
# keep it in PYBIND11_PYTEST_FILES, so that we get the "eigen is not installed"
# skip message).
list(FIND PYBIND11_TEST_FILES test_eigen.cpp PYBIND11_TEST_FILES_EIGEN_I)
if(PYBIND11_TEST_FILES_EIGEN_I GREATER -1)
  # Try loading via newer Eigen's Eigen3Config first (bypassing tools/FindEigen3.cmake).
  # Eigen 3.3.1+ exports a cmake 3.0+ target for handling dependency requirements, but also
  # produces a fatal error if loaded from a pre-3.0 cmake.
  if(DOWNLOAD_EIGEN)
    if(CMAKE_VERSION VERSION_LESS 3.11)
      message(FATAL_ERROR "CMake 3.11+ required when using DOWNLOAD_EIGEN")
    endif()

    set(EIGEN3_VERSION_STRING "3.3.8")

    include(FetchContent)
    FetchContent_Declare(
      eigen
      GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
      GIT_TAG ${EIGEN3_VERSION_STRING})

    FetchContent_GetProperties(eigen)
    if(NOT eigen_POPULATED)
      message(STATUS "Downloading Eigen")
      FetchContent_Populate(eigen)
    endif()

    set(EIGEN3_INCLUDE_DIR ${eigen_SOURCE_DIR})
    set(EIGEN3_FOUND TRUE)

  else()
    find_package(Eigen3 3.2.7 QUIET CONFIG)

    if(NOT EIGEN3_FOUND)
      # Couldn't load via target, so fall back to allowing module mode finding, which will pick up
      # tools/FindEigen3.cmake
      find_package(Eigen3 3.2.7 QUIET)
    endif()
  endif()

  if(EIGEN3_FOUND)
    if(NOT TARGET Eigen3::Eigen)
      add_library(Eigen3::Eigen IMPORTED INTERFACE)
      set_property(TARGET Eigen3::Eigen PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                                 "${EIGEN3_INCLUDE_DIR}")
    endif()

    # Eigen 3.3.1+ cmake sets EIGEN3_VERSION_STRING (and hard codes the version when installed
    # rather than looking it up in the cmake script); older versions, and the
    # tools/FindEigen3.cmake, set EIGEN3_VERSION instead.
    if(NOT EIGEN3_VERSION AND EIGEN3_VERSION_STRING)
      set(EIGEN3_VERSION ${EIGEN3_VERSION_STRING})
    endif()
    message(STATUS "Building tests with Eigen v${EIGEN3_VERSION}")
  else()
    list(REMOVE_AT PYBIND11_TEST_FILES ${PYBIND11_TEST_FILES_EIGEN_I})
    message(
      STATUS "Building tests WITHOUT Eigen, use -DDOWNLOAD_EIGEN=ON on CMake 3.11+ to download")
  endif()
endif()

# Optional dependency for some tests (boost::variant is only supported with version >= 1.56)
find_package(Boost 1.56)

if(Boost_FOUND)
  if(NOT TARGET Boost::headers)
    add_library(Boost::headers IMPORTED INTERFACE)
    if(TARGET Boost::boost)
      # Classic FindBoost
      set_property(TARGET Boost::boost PROPERTY INTERFACE_LINK_LIBRARIES Boost::boost)
    else()
      # Very old FindBoost, or newer Boost than CMake in older CMakes
      set_property(TARGET Boost::headers PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                                  ${Boost_INCLUDE_DIRS})
    endif()
  endif()
endif()

# Check if we need to add -lstdc++fs or -lc++fs or nothing
if(MSVC)
  set(STD_FS_NO_LIB_NEEDED TRUE)
else()
  file(
    WRITE ${CMAKE_CURRENT_BINARY_DIR}/main.cpp
    "#include <filesystem>\nint main(int argc, char ** argv) {\n  std::filesystem::path p(argv[0]);\n  return p.string().length();\n}"
  )
  try_compile(
    STD_FS_NO_LIB_NEEDED ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES ${CMAKE_CURRENT_BINARY_DIR}/main.cpp
    COMPILE_DEFINITIONS -std=c++17)
  try_compile(
    STD_FS_NEEDS_STDCXXFS ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES ${CMAKE_CURRENT_BINARY_DIR}/main.cpp
    COMPILE_DEFINITIONS -std=c++17
    LINK_LIBRARIES stdc++fs)
  try_compile(
    STD_FS_NEEDS_CXXFS ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES ${CMAKE_CURRENT_BINARY_DIR}/main.cpp
    COMPILE_DEFINITIONS -std=c++17
    LINK_LIBRARIES c++fs)
endif()

if(${STD_FS_NEEDS_STDCXXFS})
  set(STD_FS_LIB stdc++fs)
elseif(${STD_FS_NEEDS_CXXFS})
  set(STD_FS_LIB c++fs)
elseif(${STD_FS_NO_LIB_NEEDED})
  set(STD_FS_LIB "")
else()
  message(WARNING "Unknown compiler - not passing -lstdc++fs")
  set(STD_FS_LIB "")
endif()

# Compile with compiler warnings turned on
function(pybind11_enable_warnings target_name)
  if(MSVC)
    target_compile_options(${target_name} PRIVATE /W4)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Intel|Clang)" AND NOT PYBIND11_CUDA_TESTS)
    target_compile_options(
      ${target_name}
      PRIVATE -Wall
              -Wextra
              -Wconversion
              -Wcast-qual
              -Wdeprecated
              -Wundef
              -Wnon-virtual-dtor)
  endif()

  if(PYBIND11_WERROR)
    if(MSVC)
      target_compile_options(${target_name} PRIVATE /WX)
    elseif(PYBIND11_CUDA_TESTS)
      target_compile_options(${target_name} PRIVATE "SHELL:-Werror all-warnings")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang|IntelLLVM)")
      target_compile_options(${target_name} PRIVATE -Werror)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
      target_compile_options(
        ${target_name}
        PRIVATE
          -Werror-all
          # "Inlining inhibited by limit max-size", "Inlining inhibited by limit max-total-size"
          -diag-disable 11074,11076)
    endif()
  endif()

  # Needs to be re-added since the ordering requires these to be after the ones above
  if(CMAKE_CXX_STANDARD
     AND CMAKE_CXX_COMPILER_ID MATCHES "Clang"
     AND PYTHON_VERSION VERSION_LESS 3.0)
    if(CMAKE_CXX_STANDARD LESS 17)
      target_compile_options(${target_name} PUBLIC -Wno-deprecated-register)
    else()
      target_compile_options(${target_name} PUBLIC -Wno-register)
    endif()
  endif()
endfunction()

set(test_targets pybind11_tests)

# Build pybind11_cross_module_tests if any test_whatever.py are being built that require it
foreach(t ${PYBIND11_CROSS_MODULE_TESTS})
  list(FIND PYBIND11_PYTEST_FILES ${t} i)
  if(i GREATER -1)
    list(APPEND test_targets pybind11_cross_module_tests)
    break()
  endif()
endforeach()

foreach(t ${PYBIND11_CROSS_MODULE_GIL_TESTS})
  list(FIND PYBIND11_PYTEST_FILES ${t} i)
  if(i GREATER -1)
    list(APPEND test_targets cross_module_gil_utils)
    break()
  endif()
endforeach()

# Support CUDA testing by forcing the target file to compile with NVCC
if(PYBIND11_CUDA_TESTS)
  set_property(SOURCE ${PYBIND11_TEST_FILES} PROPERTY LANGUAGE CUDA)
endif()

foreach(target ${test_targets})
  set(test_files ${PYBIND11_TEST_FILES})
  if(NOT "${target}" STREQUAL "pybind11_tests")
    set(test_files "")
  endif()

  # Support CUDA testing by forcing the target file to compile with NVCC
  if(PYBIND11_CUDA_TESTS)
    set_property(SOURCE ${target}.cpp PROPERTY LANGUAGE CUDA)
  endif()

  # Create the binding library
  pybind11_add_module(${target} THIN_LTO ${target}.cpp ${test_files} ${PYBIND11_HEADERS})
  pybind11_enable_warnings(${target})

  if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
    get_property(
      suffix
      TARGET ${target}
      PROPERTY SUFFIX)
    set(source_output "${CMAKE_CURRENT_SOURCE_DIR}/${target}${suffix}")
    if(suffix AND EXISTS "${source_output}")
      message(WARNING "Output file also in source directory; "
                      "please remove to avoid confusion: ${source_output}")
    endif()
  endif()

  if(MSVC)
    target_compile_options(${target} PRIVATE /utf-8)
  endif()

  if(EIGEN3_FOUND)
    target_link_libraries(${target} PRIVATE Eigen3::Eigen)
    target_compile_definitions(${target} PRIVATE -DPYBIND11_TEST_EIGEN)
  endif()

  if(Boost_FOUND)
    target_link_libraries(${target} PRIVATE Boost::headers)
    target_compile_definitions(${target} PRIVATE -DPYBIND11_TEST_BOOST)
  endif()

  target_link_libraries(${target} PRIVATE ${STD_FS_LIB})

  # Always write the output file directly into the 'tests' directory (even on MSVC)
  if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                               "${CMAKE_CURRENT_BINARY_DIR}")

    if(DEFINED CMAKE_CONFIGURATION_TYPES)
      foreach(config ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${config} config)
        set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${config}
                                                   "${CMAKE_CURRENT_BINARY_DIR}")
      endforeach()
    endif()
  endif()
endforeach()

# Make sure pytest is found or produce a warning
pybind11_find_import(pytest VERSION 3.1)

if(NOT CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
  # This is not used later in the build, so it's okay to regenerate each time.
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/pytest.ini" "${CMAKE_CURRENT_BINARY_DIR}/pytest.ini"
                 COPYONLY)
  file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/pytest.ini"
       "\ntestpaths = \"${CMAKE_CURRENT_SOURCE_DIR}\"")

endif()

# cmake 3.12 added list(transform <list> prepend
# but we can't use it yet
string(REPLACE "test_" "${CMAKE_CURRENT_SOURCE_DIR}/test_" PYBIND11_ABS_PYTEST_FILES
               "${PYBIND11_PYTEST_FILES}")

set(PYBIND11_TEST_PREFIX_COMMAND
    ""
    CACHE STRING "Put this before pytest, use for checkers and such")

# A single command to compile and run the tests
add_custom_target(
  pytest
  COMMAND ${PYBIND11_TEST_PREFIX_COMMAND} ${PYTHON_EXECUTABLE} -m pytest
          ${PYBIND11_ABS_PYTEST_FILES}
  DEPENDS ${test_targets}
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  USES_TERMINAL)

if(PYBIND11_TEST_OVERRIDE)
  add_custom_command(
    TARGET pytest
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo
            "Note: not all tests run: -DPYBIND11_TEST_OVERRIDE is in effect")
endif()

# cmake-format: off
add_custom_target(
  memcheck
  COMMAND
    PYTHONMALLOC=malloc
    valgrind
    --leak-check=full
    --show-leak-kinds=definite,indirect
    --errors-for-leak-kinds=definite,indirect
    --error-exitcode=1
    --read-var-info=yes
    --track-origins=yes
    --suppressions="${CMAKE_CURRENT_SOURCE_DIR}/valgrind-python.supp"
    --suppressions="${CMAKE_CURRENT_SOURCE_DIR}/valgrind-numpy-scipy.supp"
    --gen-suppressions=all
    ${PYTHON_EXECUTABLE} -m pytest ${PYBIND11_ABS_PYTEST_FILES}
  DEPENDS ${test_targets}
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  USES_TERMINAL)
# cmake-format: on

# Add a check target to run all the tests, starting with pytest (we add dependencies to this below)
add_custom_target(check DEPENDS pytest)

# The remaining tests only apply when being built as part of the pybind11 project, but not if the
# tests are being built independently.
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  return()
endif()

# Add a post-build comment to show the primary test suite .so size and, if a previous size, compare it:
add_custom_command(
  TARGET pybind11_tests
  POST_BUILD
  COMMAND
    ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../tools/libsize.py
    $<TARGET_FILE:pybind11_tests>
    ${CMAKE_CURRENT_BINARY_DIR}/sosize-$<TARGET_FILE_NAME:pybind11_tests>.txt)

if(NOT PYBIND11_CUDA_TESTS)
  # Test embedding the interpreter. Provides the `cpptest` target.
  add_subdirectory(test_embed)

  # Test CMake build using functions and targets from subdirectory or installed location
  add_subdirectory(test_cmake_build)
endif()
