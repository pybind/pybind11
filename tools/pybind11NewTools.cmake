# tools/pybind11NewTools.cmake -- Build system for the pybind11 modules
#
# Copyright (c) 2020 Wenzel Jakob <wenzel@inf.ethz.ch> and Henry Schreiner
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

include_guard(DIRECTORY)

get_property(
  is_config
  TARGET pybind11::headers
  PROPERTY IMPORTED)

if(pybind11_FIND_QUIETLY)
  set(_pybind11_quiet QUIET)
else()
  set(_pybind11_quiet "")
endif()

if(NOT Python_FOUND AND NOT Python3_FOUND)
  if(NOT DEFINED Python_FIND_IMPLEMENTATIONS)
    set(Python_FIND_IMPLEMENTATIONS CPython PyPy)
  endif()

  # GitHub Actions like activation
  if(NOT DEFINED Python_ROOT_DIR AND DEFINED ENV{pythonLocation})
    set(Python_ROOT_DIR "$ENV{pythonLocation}")
  endif()

  # Interpreter should not be found when cross-compiling
  if(_PYBIND11_CROSSCOMPILING)
    set(_pybind11_interp_component "")
  else()
    set(_pybind11_interp_component Interpreter)
  endif()

  # Development.Module support (required for manylinux) started in 3.18;
  # Development.SABIModule (stable ABI, see the STABLE_ABI option) in 3.26.
  if(CMAKE_VERSION VERSION_LESS 3.18)
    set(_pybind11_dev_component Development)
  elseif(CMAKE_VERSION VERSION_LESS 3.26)
    set(_pybind11_dev_component Development.Module OPTIONAL_COMPONENTS Development.Embed)
  else()
    set(_pybind11_dev_component Development.Module OPTIONAL_COMPONENTS Development.Embed
                                Development.SABIModule)
  endif()

  # Callers need to be able to access Python_EXECUTABLE
  set(_pybind11_global_keyword "")
  if(NOT is_config AND NOT DEFINED Python_ARTIFACTS_INTERACTIVE)
    set(Python_ARTIFACTS_INTERACTIVE TRUE)
    if(NOT CMAKE_VERSION VERSION_LESS 3.24)
      set(_pybind11_global_keyword "GLOBAL")
    endif()
  endif()

  find_package(
    Python 3.9 REQUIRED COMPONENTS ${_pybind11_interp_component} ${_pybind11_dev_component}
                                   ${_pybind11_quiet} ${_pybind11_global_keyword})

  # If we are in submodule mode, export the Python targets to global targets.
  # If this behavior is not desired, FindPython _before_ pybind11.
  if(NOT is_config
     AND Python_ARTIFACTS_INTERACTIVE
     AND _pybind11_global_keyword STREQUAL "")
    if(TARGET Python::Python)
      set_property(TARGET Python::Python PROPERTY IMPORTED_GLOBAL TRUE)
    endif()
    if(TARGET Python::Interpreter)
      set_property(TARGET Python::Interpreter PROPERTY IMPORTED_GLOBAL TRUE)
    endif()
    if(TARGET Python::Module)
      set_property(TARGET Python::Module PROPERTY IMPORTED_GLOBAL TRUE)
    endif()
    if(TARGET Python::SABIModule)
      set_property(TARGET Python::SABIModule PROPERTY IMPORTED_GLOBAL TRUE)
    endif()
  endif()

  # Explicitly export version for callers (including our own functions)
  if(NOT is_config AND Python_ARTIFACTS_INTERACTIVE)
    set(Python_VERSION
        "${Python_VERSION}"
        CACHE INTERNAL "")
    set(Python_VERSION_MAJOR
        "${Python_VERSION_MAJOR}"
        CACHE INTERNAL "")
    set(Python_VERSION_MINOR
        "${Python_VERSION_MINOR}"
        CACHE INTERNAL "")
    set(Python_VERSION_PATCH
        "${Python_VERSION_PATCH}"
        CACHE INTERNAL "")
    if(DEFINED Python_SOSABI)
      set(Python_SOSABI
          "${Python_SOSABI}"
          CACHE INTERNAL "")
    endif()
  endif()
endif()

if(Python_FOUND)
  set(_Python
      Python
      CACHE INTERNAL "" FORCE)
elseif(Python3_FOUND)
  set(_Python
      Python3
      CACHE INTERNAL "" FORCE)
endif()

if(PYBIND11_MASTER_PROJECT)
  if(${_Python}_INTERPRETER_ID MATCHES "PyPy")
    message(STATUS "PyPy ${${_Python}_PyPy_VERSION} (Py ${${_Python}_VERSION})")
  else()
    message(STATUS "${_Python} ${${_Python}_VERSION}")
  endif()
endif()

if(NOT _PYBIND11_CROSSCOMPILING AND DEFINED ${_Python}_EXECUTABLE)
  if(DEFINED PYBIND11_PYTHON_EXECUTABLE_LAST AND NOT ${_Python}_EXECUTABLE STREQUAL
                                                 PYBIND11_PYTHON_EXECUTABLE_LAST)
    # Detect changes to the Python version/binary in subsequent CMake runs, and refresh config if needed
    unset(PYTHON_IS_DEBUG CACHE)
    unset(PYTHON_MODULE_EXTENSION CACHE)
    unset(PYTHON_MODULE_DEBUG_POSTFIX CACHE)
  endif()

  set(PYBIND11_PYTHON_EXECUTABLE_LAST
      "${${_Python}_EXECUTABLE}"
      CACHE INTERNAL "Python executable during the last CMake run")

  if(NOT DEFINED PYTHON_IS_DEBUG)
    # Debug check - see https://stackoverflow.com/questions/646518/python-how-to-detect-debug-Interpreter
    execute_process(
      COMMAND "${${_Python}_EXECUTABLE}" "-c"
              "import sys; sys.exit(hasattr(sys, 'gettotalrefcount'))"
      RESULT_VARIABLE _PYTHON_IS_DEBUG)
    set(PYTHON_IS_DEBUG
        "${_PYTHON_IS_DEBUG}"
        CACHE INTERNAL "Python debug status")
  endif()

  # Get the suffix - SO is deprecated, should use EXT_SUFFIX, but this is
  # required for PyPy3 (as of 7.3.1)
  if(NOT DEFINED PYTHON_MODULE_EXTENSION OR NOT DEFINED PYTHON_MODULE_DEBUG_POSTFIX)
    execute_process(
      COMMAND
        "${${_Python}_EXECUTABLE}" "-c"
        "import sys, importlib; s = importlib.import_module('distutils.sysconfig' if sys.version_info < (3, 10) else 'sysconfig'); print(s.get_config_var('EXT_SUFFIX') or s.get_config_var('SO'))"
      OUTPUT_VARIABLE _PYTHON_MODULE_EXT_SUFFIX
      ERROR_VARIABLE _PYTHON_MODULE_EXT_SUFFIX_ERR
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(_PYTHON_MODULE_EXT_SUFFIX STREQUAL "")
      message(
        FATAL_ERROR
          "pybind11 could not query the module file extension, likely the 'distutils'"
          "package is not installed. Full error message:\n${_PYTHON_MODULE_EXT_SUFFIX_ERR}")
    endif()

    # This needs to be available for the pybind11_extension function
    if(NOT DEFINED PYTHON_MODULE_DEBUG_POSTFIX)
      get_filename_component(_PYTHON_MODULE_DEBUG_POSTFIX "${_PYTHON_MODULE_EXT_SUFFIX}" NAME_WE)
      set(PYTHON_MODULE_DEBUG_POSTFIX
          "${_PYTHON_MODULE_DEBUG_POSTFIX}"
          CACHE INTERNAL "")
    endif()

    if(NOT DEFINED PYTHON_MODULE_EXTENSION)
      get_filename_component(_PYTHON_MODULE_EXTENSION "${_PYTHON_MODULE_EXT_SUFFIX}" EXT)
      set(PYTHON_MODULE_EXTENSION
          "${_PYTHON_MODULE_EXTENSION}"
          CACHE INTERNAL "")
      if((NOT "$ENV{SETUPTOOLS_EXT_SUFFIX}" STREQUAL "")
         AND (NOT "$ENV{SETUPTOOLS_EXT_SUFFIX}" STREQUAL "${PYTHON_MODULE_EXTENSION}"))
        message(
          AUTHOR_WARNING,
          "SETUPTOOLS_EXT_SUFFIX is set to \"$ENV{SETUPTOOLS_EXT_SUFFIX}\", "
          "but the auto-calculated Python extension suffix is \"${PYTHON_MODULE_EXTENSION}\". "
          "This may cause problems when importing the Python extensions. "
          "If you are using cross-compiling Python, you may need to "
          "set PYTHON_MODULE_EXTENSION manually.")
      endif()
    endif()
  endif()
else()
  if(NOT DEFINED PYTHON_IS_DEBUG
     OR NOT DEFINED PYTHON_MODULE_EXTENSION
     OR NOT DEFINED PYTHON_MODULE_DEBUG_POSTFIX)
    include("${CMAKE_CURRENT_LIST_DIR}/pybind11GuessPythonExtSuffix.cmake")
    pybind11_guess_python_module_extension("${_Python}")
  endif()
  if(NOT DEFINED PYTHON_IS_DEBUG
     OR NOT DEFINED PYTHON_MODULE_EXTENSION
     OR NOT DEFINED PYTHON_MODULE_DEBUG_POSTFIX)
    message(
      FATAL_ERROR
        "A Python interpreter was not found, or you are cross-compiling, and the "
        "PYTHON_IS_DEBUG, PYTHON_MODULE_EXTENSION and PYTHON_MODULE_DEBUG_POSTFIX "
        "variables could not be guessed. Set these variables appropriately before "
        "loading pybind11 (e.g. in your CMake toolchain file)")
  endif()
endif()

# A debug build of Python needs Py_DEBUG to select the matching ABI.
# https://docs.python.org/3/c-api/intro.html#debugging-builds
# https://stackoverflow.com/questions/39161202/how-to-work-around-missing-pymodule-create2-in-amd64-win-python35-d-lib
if(PYTHON_IS_DEBUG)
  set_property(
    TARGET pybind11::pybind11
    APPEND
    PROPERTY INTERFACE_COMPILE_DEFINITIONS Py_DEBUG)
endif()

# Check on every access - since Python can change - do nothing in that case.

if(DEFINED ${_Python}_INCLUDE_DIRS)
  # Only add Python for build - must be added during the import for config
  # since it has to be re-discovered.
  #
  # This needs to be a target to be included after the local pybind11
  # directory, just in case there there is an installed pybind11 sitting
  # next to Python's includes. It also ensures Python is a SYSTEM library.
  add_library(pybind11::python_headers INTERFACE IMPORTED)
  set_property(
    TARGET pybind11::python_headers PROPERTY INTERFACE_INCLUDE_DIRECTORIES
                                             "$<BUILD_INTERFACE:${${_Python}_INCLUDE_DIRS}>")
  set_property(
    TARGET pybind11::pybind11
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES pybind11::python_headers)
  set(pybind11_INCLUDE_DIRS
      "${pybind11_INCLUDE_DIR}" "${${_Python}_INCLUDE_DIRS}"
      CACHE INTERNAL "Directories where pybind11 and possibly Python headers are located")
endif()

# In CMake 3.18+, you can find these separately, so include an if
if(TARGET ${_Python}::Python)
  set_property(
    TARGET pybind11::embed
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES ${_Python}::Python)
endif()

if(TARGET ${_Python}::Module)
  # On Android, older versions of CMake don't know that modules need to link against
  # libpython, so Python::Module will be an INTERFACE target with no associated library
  # files.
  get_target_property(module_target_type ${_Python}::Module TYPE)
  if(ANDROID AND module_target_type STREQUAL INTERFACE_LIBRARY)
    target_link_libraries(${_Python}::Module INTERFACE ${${_Python}_LIBRARIES})
  endif()

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

# The Py_LIMITED_API value for STABLE_ABI modules (CPython 3.12 is the minimum pybind11 supports).
set(PYBIND11_STABLE_ABI_VERSION
    "3.12"
    CACHE STRING "Python version whose stable ABI STABLE_ABI modules target (3.12 or newer)")

set(PYBIND11_ABI3T
    OFF
    CACHE BOOL "STABLE_ABI modules target abi3t (PEP 803), also from GIL-enabled Python 3.15+")

# True if the interpreter is a free-threaded build (needs the abi3t variant of the stable ABI).
function(_pybind11_python_is_free_threaded out_var)
  if(${_Python}_FREE_THREADED
     OR "${${_Python}_SOABI}" MATCHES "^cpython-[0-9]+t"
     OR "${PYTHON_MODULE_EXTENSION}" MATCHES "^\\.cpython-[0-9]+t")
    set(${out_var}
        ON
        PARENT_SCOPE)
  else()
    set(${out_var}
        OFF
        PARENT_SCOPE)
  endif()
endfunction()

# True if STABLE_ABI modules target abi3t: always on free-threaded Python, else PYBIND11_ABI3T.
function(_pybind11_stable_abi_is_abi3t out_var)
  _pybind11_python_is_free_threaded(_abi3t)
  if(PYBIND11_ABI3T)
    set(_abi3t ON)
  endif()
  set(${out_var}
      ${_abi3t}
      PARENT_SCOPE)
endfunction()

# PYBIND11_STABLE_ABI_VERSION, raised to 3.15 for abi3t (abi3t starts there).
function(_pybind11_stable_abi_version out_var)
  set(_version "${PYBIND11_STABLE_ABI_VERSION}")
  _pybind11_stable_abi_is_abi3t(_abi3t)
  if(_abi3t AND _version VERSION_LESS 3.15)
    set(_version "3.15")
  endif()
  set(${out_var}
      "${_version}"
      PARENT_SCOPE)
endfunction()

# The effective stable ABI version as the Py_LIMITED_API hex value.
function(_pybind11_stable_abi_hex out_var)
  _pybind11_stable_abi_version(_version)
  string(REPLACE "." ";" _parts "${_version}")
  list(GET _parts 0 _major)
  list(GET _parts 1 _minor)
  math(EXPR _hex "(${_major} << 24) | (${_minor} << 16)" OUTPUT_FORMAT HEXADECIMAL)
  set(${out_var}
      "${_hex}"
      PARENT_SCOPE)
endfunction()

# Add the stable ABI define and import library to a target that FindPython's USE_SABI did not set up.
function(_pybind11_stable_abi_setup target_name)
  _pybind11_stable_abi_hex(_hex)
  _pybind11_stable_abi_is_abi3t(_abi3t)
  _pybind11_python_is_free_threaded(_free_threaded)
  if(NOT _abi3t)
    target_compile_definitions(${target_name} PRIVATE "Py_LIMITED_API=${_hex}")
  else()
    target_compile_definitions(${target_name} PRIVATE "Py_TARGET_ABI3T=${_hex}")
  endif()
  if(_abi3t
     AND NOT _free_threaded
     AND WIN32)
    # Python::SABIModule is python3.lib here, but abi3t needs python3t.lib from the same directory.
    target_include_directories(${target_name} SYSTEM PRIVATE ${${_Python}_INCLUDE_DIRS})
    find_library(
      _pybind11_python3t_lib python3t
      PATHS ${${_Python}_SABI_LIBRARY_DIRS}
      NO_DEFAULT_PATH REQUIRED)
    target_link_libraries(${target_name} PRIVATE "${_pybind11_python3t_lib}")
  else()
    target_link_libraries(${target_name} PRIVATE ${_Python}::SABIModule)
  endif()
endfunction()

# Fail with an explanation if a STABLE_ABI module cannot be built in this configuration.
function(_pybind11_check_stable_abi target_name lib_type)
  if(CMAKE_VERSION VERSION_LESS 3.26)
    message(FATAL_ERROR "${target_name}: STABLE_ABI requires CMake 3.26 or newer (USE_SABI).")
  endif()
  if(lib_type STREQUAL "STATIC")
    message(FATAL_ERROR "${target_name}: STABLE_ABI is not supported for STATIC libraries.")
  endif()
  if(PYBIND11_STABLE_ABI_VERSION VERSION_LESS 3.12)
    message(FATAL_ERROR "${target_name}: PYBIND11_STABLE_ABI_VERSION must be 3.12 or newer, "
                        "got ${PYBIND11_STABLE_ABI_VERSION}.")
  endif()
  if(DEFINED ${_Python}_INTERPRETER_ID AND NOT "${${_Python}_INTERPRETER_ID}" STREQUAL "Python")
    message(FATAL_ERROR "${target_name}: STABLE_ABI requires CPython, found "
                        "${${_Python}_INTERPRETER_ID}.")
  endif()
  if(DEFINED ${_Python}_VERSION AND ${_Python}_VERSION VERSION_LESS PYBIND11_STABLE_ABI_VERSION)
    message(FATAL_ERROR "${target_name}: STABLE_ABI ${PYBIND11_STABLE_ABI_VERSION} needs Python "
                        ">= ${PYBIND11_STABLE_ABI_VERSION} headers, found ${${_Python}_VERSION}.")
  endif()
  _pybind11_stable_abi_is_abi3t(_abi3t)
  if(_abi3t
     AND DEFINED ${_Python}_VERSION
     AND ${_Python}_VERSION VERSION_LESS 3.15)
    message(FATAL_ERROR "${target_name}: the stable ABI for free-threading (abi3t) needs Python "
                        ">= 3.15, found ${${_Python}_VERSION}.")
  endif()
  if(NOT TARGET ${_Python}::SABIModule)
    message(
      FATAL_ERROR
        "${target_name}: STABLE_ABI needs the Development.SABIModule component "
        "of FindPython. If your project calls find_package(Python) itself, add "
        "Development.SABIModule to its components.")
  endif()
endfunction()

# WITHOUT_SOABI and WITH_SOABI will disable the custom extension handling used by pybind11.
# WITH_SOABI is passed on to python_add_library.
function(pybind11_add_module target_name)
  cmake_parse_arguments(
    PARSE_ARGV
    1
    ARG
    "STATIC;SHARED;MODULE;THIN_LTO;OPT_SIZE;NO_EXTRAS;WITHOUT_SOABI;PRECOMPILE;NO_PRECOMPILE;STABLE_ABI;NO_STABLE_ABI"
    ""
    "")

  if(ARG_STATIC)
    set(lib_type STATIC)
  elseif(ARG_SHARED)
    set(lib_type SHARED)
  else()
    set(lib_type MODULE)
  endif()

  # STABLE_ABI keyword, or the PYBIND11_STABLE_ABI variable as the default; NO_STABLE_ABI opts out.
  set(stable_abi OFF)
  set(use_sabi "")
  set(own_sabi OFF)
  if((ARG_STABLE_ABI OR PYBIND11_STABLE_ABI) AND NOT ARG_NO_STABLE_ABI)
    _pybind11_check_stable_abi(${target_name} ${lib_type})
    set(stable_abi ON)
    _pybind11_stable_abi_is_abi3t(_abi3t)
    _pybind11_python_is_free_threaded(_free_threaded)
    if(lib_type STREQUAL "SHARED" OR (_abi3t AND NOT _free_threaded))
      # python_add_library(SHARED) links the embedding library, and USE_SABI only selects abi3t
      # on free-threaded Python.
      set(own_sabi ON)
    elseif(lib_type STREQUAL "MODULE")
      # Defines Py_LIMITED_API and links Python::SABIModule (python3.lib on Windows).
      _pybind11_stable_abi_version(_sabi_version)
      set(use_sabi USE_SABI ${_sabi_version})
    endif()
  endif()

  if(own_sabi)
    add_library(${target_name} ${lib_type} ${ARG_UNPARSED_ARGUMENTS})
    _pybind11_stable_abi_setup(${target_name})
  elseif("${_Python}" STREQUAL "Python")
    python_add_library(${target_name} ${lib_type} ${use_sabi} ${ARG_UNPARSED_ARGUMENTS})
  elseif("${_Python}" STREQUAL "Python3")
    python3_add_library(${target_name} ${lib_type} ${use_sabi} ${ARG_UNPARSED_ARGUMENTS})
  else()
    message(FATAL_ERROR "Cannot detect FindPython version: ${_Python}")
  endif()

  target_link_libraries(${target_name} PRIVATE pybind11::headers)

  if(lib_type STREQUAL "MODULE")
    if(NOT own_sabi)
      target_link_libraries(${target_name} PRIVATE pybind11::module)
    else()
      # Python::Module would link the version-specific library; take only the module link flags.
      target_link_libraries(${target_name} PRIVATE pybind11::python_link_helper)
    endif()
  elseif(stable_abi)
    # A SHARED helper library that stable-ABI modules link: same ABI, no embedding.
  else()
    target_link_libraries(${target_name} PRIVATE pybind11::embed)
  endif()

  _pybind11_maybe_precompile(${target_name} "${ARG_PRECOMPILE}" "${ARG_NO_PRECOMPILE}"
                             "${stable_abi}")

  _pybind11_default_hidden_visibility(${target_name})

  # If we don't pass a WITH_SOABI or WITHOUT_SOABI, use our own default handling of extensions
  if(NOT ARG_WITHOUT_SOABI AND NOT "WITH_SOABI" IN_LIST ARG_UNPARSED_ARGUMENTS)
    if(stable_abi)
      pybind11_extension_stable_abi(${target_name})
    else()
      pybind11_extension(${target_name})
    endif()
  endif()

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

  if(DEFINED CMAKE_BUILD_TYPE) # see https://github.com/pybind/pybind11/issues/4454
    # Use case-insensitive comparison to match the result of $<CONFIG:cfgs>
    string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
    if(NOT MSVC AND NOT "${uppercase_CMAKE_BUILD_TYPE}" MATCHES DEBUG|RELWITHDEBINFO|NONE)
      # Strip unnecessary sections of the binary on Linux/macOS
      pybind11_strip(${target_name})
    endif()
  endif()

  if(MSVC)
    target_link_libraries(${target_name} PRIVATE pybind11::windows_extras)
  endif()

  if(ARG_OPT_SIZE)
    target_link_libraries(${target_name} PRIVATE pybind11::opt_size)
  endif()
endfunction()

function(pybind11_extension name)
  # The extension is precomputed
  set_target_properties(
    ${name}
    PROPERTIES PREFIX ""
               DEBUG_POSTFIX "${PYTHON_MODULE_DEBUG_POSTFIX}"
               SUFFIX "${PYTHON_MODULE_EXTENSION}")
endfunction()

# Stable-ABI modules carry the version-independent "abi3" or "abi3t" tag (none on Windows).
function(pybind11_extension_stable_abi name)
  if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(_ext ".pyd")
  else()
    set(_ext "${CMAKE_SHARED_MODULE_SUFFIX}")
  endif()
  if(DEFINED ${_Python}_SOSABI AND NOT "${${_Python}_SOSABI}" STREQUAL "")
    set(_sosabi "${${_Python}_SOSABI}")
    _pybind11_stable_abi_is_abi3t(_abi3t)
    if(_abi3t AND _sosabi MATCHES "^abi3(-|$)")
      # SOSABI is abi3 on GIL-enabled Python.
      string(REGEX REPLACE "^abi3" "abi3t" _sosabi "${_sosabi}")
    endif()
    set(_ext ".${_sosabi}${_ext}")
  endif()
  set_target_properties(${name} PROPERTIES PREFIX "" SUFFIX "${_ext}")
endfunction()
