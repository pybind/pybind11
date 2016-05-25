# CMakeLists.txt -- Build system for the pybind11 examples
#
# Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 2.8.12)

project(pybind11)

# Check if pybind11 is being used directly or via add_subdirectory
set(MASTER_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(MASTER_PROJECT ON)
endif ()

option(PYBIND11_INSTALL "Install pybind11 header files?" ${MASTER_PROJECT})
option(PYBIND11_TEST "Build tests?" ${MASTER_PROJECT})

# Add a CMake parameter for choosing a desired Python version
set(PYBIND11_PYTHON_VERSION "" CACHE STRING "Python version to use for compiling the example application")

include(CheckCXXCompilerFlag)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/tools")
set(Python_ADDITIONAL_VERSIONS 3.4 3.5 3.6 3.7)
find_package(PythonLibsNew ${PYBIND11_PYTHON_VERSION} REQUIRED)

# Cache variables so pybind11_add_module can be used in parent projects
set(PYBIND11_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include" CACHE INTERNAL "")
set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} CACHE INTERNAL "")
set(PYTHON_LIBRARIES ${PYTHON_LIBRARIES} CACHE INTERNAL "")
set(PYTHON_MODULE_PREFIX ${PYTHON_MODULE_PREFIX} CACHE INTERNAL "")
set(PYTHON_MODULE_EXTENSION ${PYTHON_MODULE_EXTENSION} CACHE INTERNAL "")

# Build a Python extension module:
# pybind11_add_module(<name> source1 [source2 ...])
#
function(pybind11_add_module target_name)
  add_library(${target_name} MODULE ${ARGN})
  target_include_directories(${target_name} PUBLIC ${PYBIND11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})

  # The prefix and extension are provided by FindPythonLibsNew.cmake
  set_target_properties(${target_name} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
  set_target_properties(${target_name} PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")

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
  if(MSVC)
    target_link_libraries(${target_name} ${PYTHON_LIBRARIES})
  elseif(APPLE)
    # Make sure OS X does not have any issues with missing symbols
    set_target_properties(${target_name} PROPERTIES MACOSX_RPATH ".")
    target_link_libraries(${target_name} PRIVATE "-undefined dynamic_lookup")
  endif()

  # Make sure C++11/14 are enabled
  if(NOT MSVC)
    check_cxx_compiler_flag("-std=c++14" HAS_CPP14_FLAG)
    check_cxx_compiler_flag("-std=c++11" HAS_CPP11_FLAG)

    if (HAS_CPP14_FLAG)
      target_compile_options(${target_name} PUBLIC "-std=c++14")
    elseif (HAS_CPP11_FLAG)
      target_compile_options(${target_name} PUBLIC "-std=c++11")
    else()
      message(FATAL_ERROR "Unsupported compiler -- pybind11 requires C++11 support!")
    endif()

    string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)
    if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
      # Default symbol visibility
      target_compile_options(${target_name} PRIVATE "-fvisibility=hidden")
    endif()
  endif()
endfunction()

if (PYBIND11_TEST)
  enable_testing()
  add_subdirectory(example)
endif()

if (PYBIND11_INSTALL)
  set(PYBIND11_HEADERS
    include/pybind11/attr.h
    include/pybind11/cast.h
    include/pybind11/common.h
    include/pybind11/complex.h
    include/pybind11/descr.h
    include/pybind11/eigen.h
    include/pybind11/functional.h
    include/pybind11/numpy.h
    include/pybind11/operators.h
    include/pybind11/pybind11.h
    include/pybind11/pytypes.h
    include/pybind11/stl.h
    include/pybind11/stl_bind.h
    include/pybind11/typeid.h
  )

  install(FILES ${PYBIND11_HEADERS} DESTINATION include/pybind11)
endif()
