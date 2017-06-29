# - Find python libraries
# This module finds the libraries corresponding to the Python interpeter
# FindPythonInterp provides.
# This code sets the following variables:
#
#  PYTHONLIBS_FOUND           - have the Python libs been found
#  PYTHON_PREFIX              - path to the Python installation
#  PYTHON_LIBRARIES           - path to the python library
#  PYTHON_INCLUDE_DIRS        - path to where Python.h is found
#  PYTHON_MODULE_EXTENSION    - lib extension, e.g. '.so' or '.pyd'
#  PYTHON_MODULE_PREFIX       - lib name prefix: usually an empty string
#  PYTHON_SITE_PACKAGES       - path to installation site-packages
#  PYTHON_IS_DEBUG            - whether the Python interpreter is a debug build
#
# Thanks to talljimbo for the patch adding the 'LDVERSION' config
# variable usage.

#=============================================================================
# Copyright 2001-2009 Kitware, Inc.
# Copyright 2012 Continuum Analytics, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# * Neither the names of Kitware, Inc., the Insight Software Consortium,
# nor the names of their contributors may be used to endorse or promote
# products derived from this software without specific prior written
# permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# # A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

# Checking for the extension makes sure that `LibsNew` was found and not just `Libs`.
if(PYTHONLIBS_FOUND AND PYTHON_MODULE_EXTENSION)
    return()
endif()

# Use the Python interpreter to find the libs.
if(PythonLibsNew_FIND_REQUIRED)
    find_package(PythonInterp ${PythonLibsNew_FIND_VERSION} REQUIRED)
else()
    find_package(PythonInterp ${PythonLibsNew_FIND_VERSION})
endif()

if(NOT PYTHONINTERP_FOUND)
    set(PYTHONLIBS_FOUND FALSE)
    return()
endif()

# According to http://stackoverflow.com/questions/646518/python-how-to-detect-debug-interpreter
# testing whether sys has the gettotalrefcount function is a reliable, cross-platform
# way to detect a CPython debug interpreter.
#
# The library suffix is from the config var LDVERSION sometimes, otherwise
# VERSION. VERSION will typically be like "2.7" on unix, and "27" on windows.
execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
    "from distutils import sysconfig as s;import sys;import struct;
print('.'.join(str(v) for v in sys.version_info));
print(sys.prefix);
print(s.get_python_inc(plat_specific=True));
print(s.get_python_lib(plat_specific=True));
print(s.get_config_var('SO'));
print(hasattr(sys, 'gettotalrefcount')+0);
print(struct.calcsize('@P'));
print(s.get_config_var('LDVERSION') or s.get_config_var('VERSION'));
print(s.get_config_var('LIBDIR') or '');
print(s.get_config_var('MULTIARCH') or '');
"
    RESULT_VARIABLE _PYTHON_SUCCESS
    OUTPUT_VARIABLE _PYTHON_VALUES
    ERROR_VARIABLE _PYTHON_ERROR_VALUE)

if(NOT _PYTHON_SUCCESS MATCHES 0)
    if(PythonLibsNew_FIND_REQUIRED)
        message(FATAL_ERROR
            "Python config failure:\n${_PYTHON_ERROR_VALUE}")
    endif()
    set(PYTHONLIBS_FOUND FALSE)
    return()
endif()

# Convert the process output into a list
string(REGEX REPLACE ";" "\\\\;" _PYTHON_VALUES ${_PYTHON_VALUES})
string(REGEX REPLACE "\n" ";" _PYTHON_VALUES ${_PYTHON_VALUES})
list(GET _PYTHON_VALUES 0 _PYTHON_VERSION_LIST)
list(GET _PYTHON_VALUES 1 PYTHON_PREFIX)
list(GET _PYTHON_VALUES 2 PYTHON_INCLUDE_DIR)
list(GET _PYTHON_VALUES 3 PYTHON_SITE_PACKAGES)
list(GET _PYTHON_VALUES 4 PYTHON_MODULE_EXTENSION)
list(GET _PYTHON_VALUES 5 PYTHON_IS_DEBUG)
list(GET _PYTHON_VALUES 6 PYTHON_SIZEOF_VOID_P)
list(GET _PYTHON_VALUES 7 PYTHON_LIBRARY_SUFFIX)
list(GET _PYTHON_VALUES 8 PYTHON_LIBDIR)
list(GET _PYTHON_VALUES 9 PYTHON_MULTIARCH)

# Make sure the Python has the same pointer-size as the chosen compiler
# Skip if CMAKE_SIZEOF_VOID_P is not defined
if(CMAKE_SIZEOF_VOID_P AND (NOT "${PYTHON_SIZEOF_VOID_P}" STREQUAL "${CMAKE_SIZEOF_VOID_P}"))
    if(PythonLibsNew_FIND_REQUIRED)
        math(EXPR _PYTHON_BITS "${PYTHON_SIZEOF_VOID_P} * 8")
        math(EXPR _CMAKE_BITS "${CMAKE_SIZEOF_VOID_P} * 8")
        message(FATAL_ERROR
            "Python config failure: Python is ${_PYTHON_BITS}-bit, "
            "chosen compiler is  ${_CMAKE_BITS}-bit")
    endif()
    set(PYTHONLIBS_FOUND FALSE)
    return()
endif()

# The built-in FindPython didn't always give the version numbers
string(REGEX REPLACE "\\." ";" _PYTHON_VERSION_LIST ${_PYTHON_VERSION_LIST})
list(GET _PYTHON_VERSION_LIST 0 PYTHON_VERSION_MAJOR)
list(GET _PYTHON_VERSION_LIST 1 PYTHON_VERSION_MINOR)
list(GET _PYTHON_VERSION_LIST 2 PYTHON_VERSION_PATCH)

# Make sure all directory separators are '/'
string(REGEX REPLACE "\\\\" "/" PYTHON_PREFIX ${PYTHON_PREFIX})
string(REGEX REPLACE "\\\\" "/" PYTHON_INCLUDE_DIR ${PYTHON_INCLUDE_DIR})
string(REGEX REPLACE "\\\\" "/" PYTHON_SITE_PACKAGES ${PYTHON_SITE_PACKAGES})

if(CMAKE_HOST_WIN32)
    set(PYTHON_LIBRARY
        "${PYTHON_PREFIX}/libs/Python${PYTHON_LIBRARY_SUFFIX}.lib")

    # when run in a venv, PYTHON_PREFIX points to it. But the libraries remain in the
    # original python installation. They may be found relative to PYTHON_INCLUDE_DIR.
    if(NOT EXISTS "${PYTHON_LIBRARY}")
        get_filename_component(_PYTHON_ROOT ${PYTHON_INCLUDE_DIR} DIRECTORY)
        set(PYTHON_LIBRARY
            "${_PYTHON_ROOT}/libs/Python${PYTHON_LIBRARY_SUFFIX}.lib")
    endif()

    # raise an error if the python libs are still not found.
    if(NOT EXISTS "${PYTHON_LIBRARY}")
        message(FATAL_ERROR "Python libraries not found")
    endif()

else()
    if(PYTHON_MULTIARCH)
        set(_PYTHON_LIBS_SEARCH "${PYTHON_LIBDIR}/${PYTHON_MULTIARCH}" "${PYTHON_LIBDIR}")
    else()
        set(_PYTHON_LIBS_SEARCH "${PYTHON_LIBDIR}")
    endif()
    #message(STATUS "Searching for Python libs in ${_PYTHON_LIBS_SEARCH}")
    # Probably this needs to be more involved. It would be nice if the config
    # information the python interpreter itself gave us were more complete.
    find_library(PYTHON_LIBRARY
        NAMES "python${PYTHON_LIBRARY_SUFFIX}"
        PATHS ${_PYTHON_LIBS_SEARCH}
        NO_DEFAULT_PATH)

    # If all else fails, just set the name/version and let the linker figure out the path.
    if(NOT PYTHON_LIBRARY)
        set(PYTHON_LIBRARY python${PYTHON_LIBRARY_SUFFIX})
    endif()
endif()

MARK_AS_ADVANCED(
  PYTHON_LIBRARY
  PYTHON_INCLUDE_DIR
)

# We use PYTHON_INCLUDE_DIR, PYTHON_LIBRARY and PYTHON_DEBUG_LIBRARY for the
# cache entries because they are meant to specify the location of a single
# library. We now set the variables listed by the documentation for this
# module.
SET(PYTHON_INCLUDE_DIRS "${PYTHON_INCLUDE_DIR}")
SET(PYTHON_LIBRARIES "${PYTHON_LIBRARY}")
SET(PYTHON_DEBUG_LIBRARIES "${PYTHON_DEBUG_LIBRARY}")

find_package_message(PYTHON
    "Found PythonLibs: ${PYTHON_LIBRARY}"
    "${PYTHON_EXECUTABLE}${PYTHON_VERSION}")

set(PYTHONLIBS_FOUND TRUE)
