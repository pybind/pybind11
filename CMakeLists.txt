# CMakeLists.txt -- Build system for the pybind11 examples
#
# Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 2.8)

project(pybind)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'MinSizeRel' as none was specified.")
  set(CMAKE_BUILD_TYPE MinSizeRel CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

set(PYBIND_PYTHON_VERSION "" CACHE STRING "Python version to use for compiling the example application")

set(Python_ADDITIONAL_VERSIONS 3.4 3.5 3.6)
find_package(PythonLibs ${PYBIND_PYTHON_VERSION} REQUIRED)
find_package(PythonInterp ${PYBIND_PYTHON_VERSION} REQUIRED)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)
if (UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -flto")
  endif()
endif()

# Compile with compiler warnings turned on
if(MSVC)
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

include_directories(${PYTHON_INCLUDE_DIR} include)

add_library(example SHARED
  include/pybind/cast.h
  include/pybind/common.h
  include/pybind/operators.h
  include/pybind/pybind.h
  include/pybind/pytypes.h
  include/pybind/typeid.h
  include/pybind/numpy.h
  example/example.cpp
  example/example1.cpp
  example/example2.cpp
  example/example3.cpp
  example/example4.cpp
  example/example5.cpp
  example/example6.cpp
  example/example7.cpp
  example/example8.cpp
  example/example9.cpp
  example/example10.cpp
  example/example11.cpp
  example/example12.cpp
)

set_target_properties(example PROPERTIES PREFIX "")
set_target_properties(example PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/example)

if (WIN32)
  if (MSVC)
    # Enforce size-based optimization and link time code generation on MSVC (~30% smaller binaries in experiments)
    # /bigobj is needed for bigger binding projects due to the limit to 64k addressable sections
    # /MP enables multithreaded builds
    set_target_properties(example PROPERTIES COMPILE_FLAGS "/Os /GL /MP /bigobj")
    set_target_properties(example PROPERTIES LINK_FLAGS "/LTCG")
  endif()

  # .PYD file extension on Windows
  set_target_properties(example PROPERTIES SUFFIX ".pyd")

  # Link against the Python shared library
  target_link_libraries(example ${PYTHON_LIBRARY})
elseif (UNIX)
  # It's quite common to have multiple copies of the same Python version
  # installed on one's system. E.g.: one copy from the OS and another copy
  # that's statically linked into an application like Blender or Maya.
  # If we link our plugin library against the OS Python here and import it
  # into Blender or Maya later on, this will cause segfaults when multiple
  # conflicting Python instances are active at the same time.

  # Windows does not seem to be affected by this issue. The solution for Linux
  # and Mac OS is simple: we just don't link against the Python library. The
  # resulting shared library will have missing symbols, but that's perfectly
  # fine -- they will be resolved at import time.

  # .SO file extension on Linux/Mac OS
  set_target_properties(example PROPERTIES SUFFIX ".so")

  # Strip unnecessary sections of the binary on Linux/Mac OS
  if(APPLE)
    set_target_properties(example PROPERTIES MACOSX_RPATH ".")
    set_target_properties(example PROPERTIES LINK_FLAGS "-undefined dynamic_lookup -dead_strip")
    if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
      add_custom_command(TARGET example POST_BUILD COMMAND strip -u -r ${PROJECT_SOURCE_DIR}/example/example.so)
    endif()
  else()
    if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
      add_custom_command(TARGET example POST_BUILD COMMAND strip ${PROJECT_SOURCE_DIR}/example/example.so)
    endif()
  endif()
endif()

enable_testing()
set(RUN_TEST ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/example/run_test.py)
foreach(i RANGE 1 12)
  add_test(NAME example${i} COMMAND ${RUN_TEST} example${i})
endforeach()
