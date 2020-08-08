# tools/pybind11NewTools.cmake -- Build system for the pybind11 modules
#
# Copyright (c) 2020 Wenzel Jakob <wenzel@inf.ethz.ch> and Henry Schreiner
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

get_property(
  is_config
  TARGET pybind11::headers
  PROPERTY IMPORTED)

if(CMAKE_VERSION VERSION_LESS 3.12)
  message(FATAL_ERROR "You cannot use the new FindPython module with CMake < 3.12")
endif()

if(NOT Python_FOUND)
  if(NOT DEFINED Python_FIND_IMPLEMENTATIONS)
    set(Python_FIND_IMPLEMENTATIONS CPython PyPy)
  endif()

  # GitHub Actions like activation
  if(NOT DEFINED Python_ROOT_DIR AND DEFINED ENV{pythonLocation})
    set(Python_ROOT_DIR "$ENV{pythonLocation}")
  endif()

  find_package(Python COMPONENTS Interpreter Development)

  # If we are in submodule mode, export the Python targets to global targets.
  # If this behavior is not desired, FindPython _before_ pybind11.
  if(NOT is_config)
    set_property(TARGET Python::Python PROPERTY IMPORTED_GLOBAL TRUE)
    set_property(TARGET Python::Interpreter PROPERTY IMPORTED_GLOBAL TRUE)
    if(TARGET Python::Module)
      set_property(TARGET Python::Module PROPERTY IMPORTED_GLOBAL TRUE)
    endif()
  endif()
endif()

# Check on every access - since Python2 and Python3 could have been used - do nothing in that case.

if(DEFINED Python_INCLUDE_DIRS)
  set_property(
    TARGET pybind11::pybind11
    APPEND
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${Python_INCLUDE_DIRS}>)
endif()

if(DEFINED Python_VERSION AND Python_VERSION VERSION_LESS 3)
  set_property(
    TARGET pybind11::pybind11
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES pybind11::python2_no_register)
endif()

# In CMake 3.18+, you can find these separately, so include an if
if(TARGET Python::Python)
  set_property(
    TARGET pybind11::embed
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES Python::Python)
endif()

# CMake 3.15+ has this
if(TARGET Python::Module)
  set_property(
    TARGET pybind11::module
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES Python::Module)
else()
  set_property(
    TARGET pybind11::module
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES pybind11::python_link_helper)
endif()

function(pybind11_add_module target_name)
  cmake_parse_arguments(PARSE_ARGV 1 ARG "STATIC;SHARED;MODULE;THIN_LTO;NO_EXTRAS" "" "")

  if(ARG_ADD_LIBRARY_STATIC)
    set(type STATIC)
  elseif(ARG_ADD_LIBRARY_SHARED)
    set(type SHARED)
  else()
    set(type MODULE)
  endif()

  if(COMMAND Python_add_library)
    python_add_library(${target_name} ${type} WITH_SOABI ${ARG_UNPARSED_ARGUMENTS})
  endif()

  target_link_libraries(${target_name} PRIVATE pybind11::headers)

  if(type STREQUAL "MODULE")
    target_link_libraries(${target_name} PRIVATE pybind11::module)
  else()
    target_link_libraries(${target_name} PRIVATE pybind11::embed)
  endif()

  if(MSVC)
    target_link_libraries(${target_name} PRIVATE pybind11::windows_extras)
  endif()

  if(DEFINED Python_VERSION AND Python_VERSION VERSION_LESS 3)
    target_link_libraries(${target_name} PRIVATE pybind11::python2_no_register)
  endif()

  # Currently Debug Python interps not supported for Python < 3.8

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
    # Strip unnecessary sections of the binary on Linux/Mac OS
    pybind11_strip(${target_name})
  endif()

  if(MSVC)
    target_link_libraries(${target_name} PRIVATE pybind11::windows_extras)
  endif()
endfunction()

function(pybind11_extension name)
  set_property(TARGET ${name} PROPERTY PREFIX "")

  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set_property(TARGET ${name} PROPERTY SUFFIX ".pyd")
  endif()

  if(Python_SOABI)
    get_property(
      suffix
      TARGET ${name}
      PROPERTY SUFFIX)
    if(NOT suffix)
      set(suffix "${CMAKE_SHARED_MODULE_SUFFIX}")
    endif()
    set_property(TARGET ${name} PROPERTY SUFFIX ".${Python_SOABI}${suffix}")
  endif()
endfunction()
