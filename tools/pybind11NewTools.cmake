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

if(pybind11_FIND_QUIETLY)
  set(_pybind11_quiet QUIET)
endif()

if(CMAKE_VERSION VERSION_LESS 3.12)
  message(FATAL_ERROR "You cannot use the new FindPython module with CMake < 3.12")
endif()

if(NOT Python_FOUND
   AND NOT Python3_FOUND
   AND NOT Python2_FOUND)
  if(NOT DEFINED Python_FIND_IMPLEMENTATIONS)
    set(Python_FIND_IMPLEMENTATIONS CPython PyPy)
  endif()

  # GitHub Actions like activation
  if(NOT DEFINED Python_ROOT_DIR AND DEFINED ENV{pythonLocation})
    set(Python_ROOT_DIR "$ENV{pythonLocation}")
  endif()

  find_package(Python REQUIRED COMPONENTS Interpreter Development ${_pybind11_quiet})

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

if(Python_FOUND)
  set(_Python
      Python
      CACHE INTERNAL "" FORCE)
elseif(Python3_FOUND AND NOT Python2_FOUND)
  set(_Python
      Python3
      CACHE INTERNAL "" FORCE)
elseif(Python2_FOUND AND NOT Python3_FOUND)
  set(_Python
      Python2
      CACHE INTERNAL "" FORCE)
else()
  message(AUTHOR_WARNING "Python2 and Python3 both present, pybind11 in "
                         "PYBIND11_NOPYTHON mode (manually activate to silence warning)")
  set(_pybind11_nopython ON)
  return()
endif()

if(PYBIND11_MASTER_PROJECT)
  if(${_Python}_INTERPRETER_ID MATCHES "PyPy")
    message(STATUS "PyPy ${${_Python}_PyPy_VERSION} (Py ${${_Python}_VERSION})")
  else()
    message(STATUS "${_Python} ${${_Python}_VERSION}")
  endif()
endif()

# Debug check - see https://stackoverflow.com/questions/646518/python-how-to-detect-debug-Interpreter
execute_process(COMMAND ${_Python}::Python -c "import sys; print(hasattr(sys, 'gettotalrefcount'))"
                OUTPUT_VARIABLE PYTHON_IS_DEBUG)

# Python debug libraries expose slightly different objects before 3.8
# https://docs.python.org/3.6/c-api/intro.html#debugging-builds
# https://stackoverflow.com/questions/39161202/how-to-work-around-missing-pymodule-create2-in-amd64-win-python35-d-lib
if(PYTHON_IS_DEBUG)
  set_property(
    TARGET pybind11::pybind11
    APPEND
    PROPERTY INTERFACE_COMPILE_DEFINITIONS Py_DEBUG)
endif()

# Check on every access - since Python2 and Python3 could have been used - do nothing in that case.

if(DEFINED ${_Python}_INCLUDE_DIRS)
  set_property(
    TARGET pybind11::pybind11
    APPEND
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${${_Python}_INCLUDE_DIRS}>)
endif()

if(DEFINED ${_Python}_VERSION AND ${_Python}_VERSION VERSION_LESS 3)
  set_property(
    TARGET pybind11::pybind11
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES pybind11::python2_no_register)
endif()

# In CMake 3.18+, you can find these separately, so include an if
if(TARGET ${_Python}::${_Python})
  set_property(
    TARGET pybind11::embed
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES ${_Python}::${_Python})
endif()

# CMake 3.15+ has this
if(TARGET ${_Python}::Module)
  set_property(
    TARGET pybind11::module
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES ${_Python}::Module)
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

  if("${_Python}" STREQUAL "Python")
    python_add_library(${target_name} ${type} WITH_SOABI ${ARG_UNPARSED_ARGUMENTS})
  elseif("${_Python}" STREQUAL "Python3")
    python3_add_library(${target_name} ${type} WITH_SOABI ${ARG_UNPARSED_ARGUMENTS})
  elseif("${_Python}" STREQUAL "Python2")
    python2_add_library(${target_name} ${type} WITH_SOABI ${ARG_UNPARSED_ARGUMENTS})
  else()
    message(FATAL_ERROR "Cannot detect FindPython version: ${_Python}")
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

  if(DEFINED ${_Python}_VERSION AND ${_Python}_VERSION VERSION_LESS 3)
    target_link_libraries(${target_name} PRIVATE pybind11::python2_no_register)
  endif()

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

  if(${_Python}_SOABI)
    get_property(
      suffix
      TARGET ${name}
      PROPERTY SUFFIX)
    if(NOT suffix)
      set(suffix "${CMAKE_SHARED_MODULE_SUFFIX}")
    endif()
    set_property(TARGET ${name} PROPERTY SUFFIX ".${${_Python}_SOABI}${suffix}")
  endif()
endfunction()
