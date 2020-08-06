# tools/pybind11Tools.cmake -- Build system for the pybind11 modules
#
# Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

# Add a CMake parameter for choosing a desired Python version
if(NOT PYBIND11_PYTHON_VERSION)
  set(PYBIND11_PYTHON_VERSION
      ""
      CACHE STRING "Python version to use for compiling modules")
endif()

# A user can set versions manually too
set(Python_ADDITIONAL_VERSIONS
    "3.9;3.8;3.7;3.6;3.5;3.4"
    CACHE INTERNAL "")

# Support for GitHub Actions-like activations
if(NOT DEFINED PYTHON_EXECUTABLE AND DEFINED ENV{pythonLocation})
  if(EXISTS "$ENV{pythonLocation}/bin/python")
    set(PYTHON_EXECUTABLE "$ENV{pythonLocation}/bin/python")
  elseif(EXISTS "$ENV{pythonLocation}/python")
    set(PYTHON_EXECUTABLE "$ENV{pythonLocation}/python")
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
find_package(PythonLibsNew ${PYBIND11_PYTHON_VERSION} MODULE REQUIRED)
list(REMOVE_AT CMAKE_MODULE_PATH -1)

# Warn or error if old variable name used
if(PYBIND11_CPP_STANDARD)
  string(REGEX MATCH [[..$]] VAL "${PYBIND11_CPP_STANDARD}")
  if(CMAKE_CXX_STANDARD)
    if(NOT CMAKE_CXX_STANDARD STREQUAL VAL)
      message(WARNING "CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} does not match "
                      "PYBIND11_CPP_STANDARD=${PYBIND11_CPP_STANDARD}, "
                      "please remove PYBIND11_CPP_STANDARD from your cache")
    endif()
  else()
    set(supported_standards 11 14 17 20)
    if("${VAL}" IN_LIST supported_standards)
      message(WARNING "USE -DCMAKE_CXX_STANDARD=${VAL} instead of PYBIND11_PYTHON_VERSION")
      set(CMAKE_CXX_STANDARD
          ${VAL}
          CACHE STRING "From PYBIND11_CPP_STANDARD")
    else()
      message(FATAL_ERROR "PYBIND11_CPP_STANDARD should be replaced with CMAKE_CXX_STANDARD "
                          "(last two chars: ${VAL} not understood as a valid CXX std)")
    endif()
  endif()
endif()

# Cache variables so pybind11_add_module can be used in parent projects
set(PYTHON_INCLUDE_DIRS
    ${PYTHON_INCLUDE_DIRS}
    CACHE INTERNAL "")
set(PYTHON_LIBRARIES
    ${PYTHON_LIBRARIES}
    CACHE INTERNAL "")
set(PYTHON_MODULE_PREFIX
    ${PYTHON_MODULE_PREFIX}
    CACHE INTERNAL "")
set(PYTHON_MODULE_EXTENSION
    ${PYTHON_MODULE_EXTENSION}
    CACHE INTERNAL "")
set(PYTHON_VERSION_MAJOR
    ${PYTHON_VERSION_MAJOR}
    CACHE INTERNAL "")
set(PYTHON_VERSION_MINOR
    ${PYTHON_VERSION_MINOR}
    CACHE INTERNAL "")
set(PYTHON_VERSION
    ${PYTHON_VERSION}
    CACHE INTERNAL "")
set(PYTHON_IS_DEBUG
    "${PYTHON_IS_DEBUG}"
    CACHE INTERNAL "")

# Only add Python for build - must be added during the import for config since it has to be re-discovered.
set_property(
  TARGET pybind11::pybind11
  APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${PYTHON_INCLUDE_DIRS}>)

set_property(
  TARGET pybind11::module
  APPEND
  PROPERTY
    INTERFACE_LINK_LIBRARIES pybind11::python_link_helper
    "$<$<OR:$<PLATFORM_ID:Windows>,$<PLATFORM_ID:Cygwin>>:$<BUILD_INTERFACE:${PYTHON_LIBRARIES}>>")

if(PYTHON_VERSION VERSION_LESS 3)
  set_property(
    TARGET pybind11::pybind11
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES pybind11::python2_no_register)
endif()

set_property(
  TARGET pybind11::embed
  APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES pybind11::pybind11 $<BUILD_INTERFACE:${PYTHON_LIBRARIES}>)

function(pybind11_extension name)
  # The prefix and extension are provided by FindPythonLibsNew.cmake
  set_target_properties(${name} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                           SUFFIX "${PYTHON_MODULE_EXTENSION}")
endfunction()

# Build a Python extension module:
# pybind11_add_module(<name> [MODULE | SHARED] [EXCLUDE_FROM_ALL]
#                     [NO_EXTRAS] [THIN_LTO] source1 [source2 ...])
#
function(pybind11_add_module target_name)
  set(options MODULE SHARED EXCLUDE_FROM_ALL NO_EXTRAS SYSTEM THIN_LTO)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "${options}" "" "")

  if(ARG_MODULE AND ARG_SHARED)
    message(FATAL_ERROR "Can't be both MODULE and SHARED")
  elseif(ARG_SHARED)
    set(lib_type SHARED)
  else()
    set(lib_type MODULE)
  endif()

  if(ARG_EXCLUDE_FROM_ALL)
    set(exclude_from_all EXCLUDE_FROM_ALL)
  else()
    set(exclude_from_all "")
  endif()

  add_library(${target_name} ${lib_type} ${exclude_from_all} ${ARG_UNPARSED_ARGUMENTS})

  target_link_libraries(${target_name} PRIVATE pybind11::module)

  if(ARG_SYSTEM)
    message(
      STATUS
        "Warning: this does not have an effect - use NO_SYSTEM_FROM_IMPORTED if using imported targets"
    )
  endif()

  # Python debug libraries expose slightly different objects before 3.8
  # https://docs.python.org/3.6/c-api/intro.html#debugging-builds
  # https://stackoverflow.com/questions/39161202/how-to-work-around-missing-pymodule-create2-in-amd64-win-python35-d-lib
  if(PYTHON_IS_DEBUG)
    target_compile_definitions(${target_name} PRIVATE Py_DEBUG)
  endif()

  pybind11_extension(${target_name})

  # -fvisibility=hidden is required to allow multiple modules compiled against
  # different pybind versions to work properly, and for some features (e.g.
  # py::module_local).  We force it on everything inside the `pybind11`
  # namespace; also turning it on for a pybind module compilation here avoids
  # potential warnings or issues from having mixed hidden/non-hidden types.
  set_target_properties(${target_name} PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                                  CUDA_VISIBILITY_PRESET "hidden")

  if(ARG_NO_EXTRAS)
    return()
  endif()

  if(NOT DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    if(ARG_THIN_LTO)
      target_link_libraries(${target_name} PRIVATE pybind11::thin_lto)
    else()
      target_link_libraries(${target_name} PRIVATE pybind11::lto)
    endif()
  endif()

  if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    pybind11_strip(${target_name})
  endif()

  if(MSVC)
    target_link_libraries(${target_name} PRIVATE pybind11::windows_extras)
  endif()

endfunction()
