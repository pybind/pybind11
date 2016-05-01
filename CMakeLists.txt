# CMakeLists.txt -- Build system for the pybind11 examples
#
# Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 2.8)

project(pybind11)

option(PYBIND11_INSTALL "Install pybind11 header files?" ON)

# Add a CMake parameter for choosing a desired Python version
set(PYBIND11_PYTHON_VERSION "" CACHE STRING "Python version to use for compiling the example application")

include(CheckCXXCompilerFlag)

# Set a default build configuration if none is specified. 'MinSizeRel' produces the smallest binaries
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'MinSizeRel' as none was specified.")
  set(CMAKE_BUILD_TYPE MinSizeRel CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

set(Python_ADDITIONAL_VERSIONS 3.4 3.5 3.6 3.7)
if (NOT ${PYBIND11_PYTHON_VERSION} STREQUAL "")
  find_package(PythonLibs ${PYBIND11_PYTHON_VERSION} EXACT)
  if (NOT PythonLibs_FOUND)
    find_package(PythonLibs ${PYBIND11_PYTHON_VERSION} REQUIRED)
  endif()
else()
  find_package(PythonLibs REQUIRED)
endif()
# The above sometimes returns version numbers like "3.4.3+"; the "+" must be removed for the next line to work
string(REPLACE "+" "" PYTHONLIBS_VERSION_STRING "+${PYTHONLIBS_VERSION_STRING}")
find_package(PythonInterp ${PYTHONLIBS_VERSION_STRING} EXACT REQUIRED)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  CHECK_CXX_COMPILER_FLAG("-std=c++14" HAS_CPP14_FLAG)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" HAS_CPP11_FLAG)

  if (HAS_CPP14_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
  elseif (HAS_CPP11_FLAG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  else()
    message(FATAL_ERROR "Unsupported compiler -- pybind11 requires C++11 support!")
  endif()

  # Enable link time optimization and set the default symbol
  # visibility to hidden (very important to obtain small binaries)
  if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
    # Default symbol visibility
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

    # Check for Link Time Optimization support
    # (GCC/Clang)
    CHECK_CXX_COMPILER_FLAG("-flto" HAS_LTO_FLAG)
    if (HAS_LTO_FLAG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
    endif()

    # Intel equivalent to LTO is called IPO
    if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
      CHECK_CXX_COMPILER_FLAG("-ipo" HAS_IPO_FLAG)
      if (HAS_IPO_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ipo")
      endif()
    endif()
  endif()
endif()

# Compile with compiler warnings turned on
if(MSVC)
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

# Include path for pybind11 header files
include_directories(include)

# Include path for Python header files
include_directories(${PYTHON_INCLUDE_DIR})

set(PYBIND11_HEADERS
  include/pybind11/attr.h
  include/pybind11/cast.h
  include/pybind11/common.h
  include/pybind11/complex.h
  include/pybind11/descr.h
  include/pybind11/functional.h
  include/pybind11/numpy.h
  include/pybind11/operators.h
  include/pybind11/pybind11.h
  include/pybind11/pytypes.h
  include/pybind11/stl.h
  include/pybind11/typeid.h
)

set(PYBIND11_EXAMPLES
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
  example/example13.cpp
  example/example14.cpp
  example/example15.cpp
  example/example16.cpp
  example/issues.cpp
)

# Create the binding library
add_library(example SHARED
  ${PYBIND11_HEADERS}
  example/example.cpp
  ${PYBIND11_EXAMPLES}
)

# Don't add a 'lib' prefix to the shared library
set_target_properties(example PROPERTIES PREFIX "")

# Always write the output file directly into the 'example' directory (even on MSVC)
set(CompilerFlags
  LIBRARY_OUTPUT_DIRECTORY LIBRARY_OUTPUT_DIRECTORY_RELEASE LIBRARY_OUTPUT_DIRECTORY_DEBUG
  LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO
  RUNTIME_OUTPUT_DIRECTORY RUNTIME_OUTPUT_DIRECTORY_RELEASE RUNTIME_OUTPUT_DIRECTORY_DEBUG
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO)

foreach(CompilerFlag ${CompilerFlags})
  set_target_properties(example PROPERTIES ${CompilerFlag} ${PROJECT_SOURCE_DIR}/example)
endforeach()

if (WIN32)
  if (MSVC)
    # /MP enables multithreaded builds (relevant when there are many files), /bigobj is
    # needed for bigger binding projects due to the limit to 64k addressable sections
    set_property(TARGET example APPEND PROPERTY COMPILE_OPTIONS /MP /bigobj)
    # Enforce size-based optimization and link time code generation on MSVC
    # (~30% smaller binaries in experiments); do nothing in debug mode.
    set_property(TARGET example APPEND PROPERTY COMPILE_OPTIONS
      "$<$<CONFIG:Release>:/Os>" "$<$<CONFIG:Release>:/GL>"
      "$<$<CONFIG:MinSizeRel>:/Os>" "$<$<CONFIG:MinSizeRel>:/GL>"
      "$<$<CONFIG:RelWithDebInfo>:/Os>" "$<$<CONFIG:RelWithDebInfo>:/GL>"
    )
    set_property(TARGET example APPEND_STRING PROPERTY LINK_FLAGS_RELEASE "/LTCG ")
    set_property(TARGET example APPEND_STRING PROPERTY LINK_FLAGS_MINSIZEREL "/LTCG ")
    set_property(TARGET example APPEND_STRING PROPERTY LINK_FLAGS_RELWITHDEBINFO "/LTCG ")
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
  # conflicting Python instances are active at the same time (even when they
  # are of the same version).

  # Windows is not affected by this issue since it handles DLL imports
  # differently. The solution for Linux and Mac OS is simple: we just don't
  # link against the Python library. The resulting shared library will have
  # missing symbols, but that's perfectly fine -- they will be resolved at
  # import time.

  # .SO file extension on Linux/Mac OS
  set_target_properties(example PROPERTIES SUFFIX ".so")

  # Optimize for a small binary size
  if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
    set_target_properties(example PROPERTIES COMPILE_FLAGS "-Os")
  endif()

  # Strip unnecessary sections of the binary on Linux/Mac OS
  if(APPLE)
    set_target_properties(example PROPERTIES MACOSX_RPATH ".")
    set_target_properties(example PROPERTIES LINK_FLAGS "-undefined dynamic_lookup ")
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
if (MSVC OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(RUN_TEST ${RUN_TEST} --relaxed)
endif()

foreach(VALUE ${PYBIND11_EXAMPLES})
  string(REGEX REPLACE "^example/(.+).cpp$" "\\1" EXAMPLE_NAME "${VALUE}")
  add_test(NAME ${EXAMPLE_NAME} COMMAND ${RUN_TEST} ${EXAMPLE_NAME})
endforeach()

if (PYBIND11_INSTALL)
    install(FILES ${PYBIND11_HEADERS} DESTINATION include/pybind11)
endif()
