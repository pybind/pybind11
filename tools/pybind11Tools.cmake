# tools/pybind11Tools.cmake -- Build system for the pybind11 modules
#
# Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 2.8.12)

# Add a CMake parameter for choosing a desired Python version
if(NOT PYBIND11_PYTHON_VERSION)
  set(PYBIND11_PYTHON_VERSION "" CACHE STRING "Python version to use for compiling modules")
endif()

set(Python_ADDITIONAL_VERSIONS 3.7 3.6 3.5 3.4)
find_package(PythonLibsNew ${PYBIND11_PYTHON_VERSION} REQUIRED)

include(CheckCXXCompilerFlag)
include(CMakeParseArguments)

function(select_cxx_standard)
  if(NOT MSVC AND NOT PYBIND11_CPP_STANDARD)
    check_cxx_compiler_flag("-std=c++14" HAS_CPP14_FLAG)
    check_cxx_compiler_flag("-std=c++11" HAS_CPP11_FLAG)

    if (HAS_CPP14_FLAG)
      set(PYBIND11_CPP_STANDARD -std=c++14)
    elseif (HAS_CPP11_FLAG)
      set(PYBIND11_CPP_STANDARD -std=c++11)
    else()
      message(FATAL_ERROR "Unsupported compiler -- pybind11 requires C++11 support!")
    endif()

    set(PYBIND11_CPP_STANDARD ${PYBIND11_CPP_STANDARD} CACHE STRING
        "C++ standard flag, e.g. -std=c++11 or -std=c++14. Defaults to latest available." FORCE)
  endif()
endfunction()

# Internal: try compiling with the given c++ and linker flags; set success_out to 1 if it works
function(_pybind11_check_cxx_and_linker_flags cxxflags linkerflags success_out)
  set(CMAKE_REQUIRED_QUIET 1)
  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${linkerflags})
  check_cxx_compiler_flag("${cxxflags}" has_lto)
  set(${success_out} ${has_lto} PARENT_SCOPE)
endfunction()

# Wraps the above in a macro that sets the flags in the parent scope and returns if the check
# succeeded (since this is a macro, not a function, the PARENT_SCOPE set and return is from the
# caller's POV)
macro(_pybind11_return_if_cxx_and_linker_flags_work feature cxxflags linkerflags cxx_out linker_out)
  _pybind11_check_cxx_and_linker_flags(${cxxflags} ${linkerflags} check_success)
  if(check_success)
    set(${cxx_out} ${cxxflags} PARENT_SCOPE)
    set(${linker_out} ${linkerflags} PARENT_SCOPE)
    message(STATUS "Found ${feature} flags: ${cxxflags}")
    return()
  endif()
endmacro()

# Internal: find the appropriate link time optimization flags for this compiler
function(_pybind11_find_lto_flags cxxflags_out linkerflags_out prefer_thin_lto)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(cxx_append "")
    set(linker_append "")
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT APPLE)
      # Clang Gold plugin does not support -Os; append -O3 to MinSizeRel builds to override it
      set(linker_append "$<$<CONFIG:MinSizeRel>: -O3>")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
      set(cxx_append " -fno-fat-lto-objects")
    endif()

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND ${prefer_thin_lto})
      _pybind11_return_if_cxx_and_linker_flags_work("LTO"
        "-flto=thin${cxx_append}" "-flto=thin${linker_append}" ${cxxflags_out} ${linkerflags_out})
    endif()

    _pybind11_return_if_cxx_and_linker_flags_work("LTO"
      "-flto${cxx_append}" "-flto${linker_append}" ${cxxflags_out} ${linkerflags_out})
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # Intel equivalent to LTO is called IPO
    _pybind11_return_if_cxx_and_linker_flags_work("LTO"
      "-ipo" "-ipo" ${cxxflags_out} ${linkerflags_out})
  elseif(MSVC)
    # cmake only interprets libraries as linker flags when they start with a - (otherwise it
    # converts /LTCG to \LTCG as if it was a Windows path).  Luckily MSVC supports passing flags
    # with - instead of /, even if it is a bit non-standard:
    _pybind11_return_if_cxx_and_linker_flags_work("LTO"
      "/GL" "-LTCG" ${cxxflags_out} ${linkerflags_out})
  endif()

  set(${cxxflags_out} "" PARENT_SCOPE)
  set(${linkerflags_out} "" PARENT_SCOPE)
  message(STATUS "LTO disabled (not supported by the compiler and/or linker)")
endfunction()

# Build a Python extension module:
# pybind11_add_module(<name> [MODULE | SHARED] [EXCLUDE_FROM_ALL]
#                     [NO_EXTRAS] [THIN_LTO] source1 [source2 ...])
#
function(pybind11_add_module target_name)
  set(options MODULE SHARED EXCLUDE_FROM_ALL NO_EXTRAS THIN_LTO)
  cmake_parse_arguments(ARG "${options}" "" "" ${ARGN})

  if(ARG_MODULE AND ARG_SHARED)
    message(FATAL_ERROR "Can't be both MODULE and SHARED")
  elseif(ARG_SHARED)
    set(lib_type SHARED)
  else()
    set(lib_type MODULE)
  endif()

  if(ARG_EXCLUDE_FROM_ALL)
    set(exclude_from_all EXCLUDE_FROM_ALL)
  endif()

  add_library(${target_name} ${lib_type} ${exclude_from_all} ${ARG_UNPARSED_ARGUMENTS})

  target_include_directories(${target_name}
    PRIVATE ${PYBIND11_INCLUDE_DIR}  # from project CMakeLists.txt
    PRIVATE ${pybind11_INCLUDE_DIR}  # from pybind11Config
    PRIVATE ${PYTHON_INCLUDE_DIRS})

  # The prefix and extension are provided by FindPythonLibsNew.cmake
  set_target_properties(${target_name} PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}")
  set_target_properties(${target_name} PROPERTIES SUFFIX "${PYTHON_MODULE_EXTENSION}")

  if(WIN32 OR CYGWIN)
    # Link against the Python shared library on Windows
    target_link_libraries(${target_name} PRIVATE ${PYTHON_LIBRARIES})
  elseif(APPLE)
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

    target_link_libraries(${target_name} PRIVATE "-undefined dynamic_lookup")

    if(ARG_SHARED)
      # Suppress CMake >= 3.0 warning for shared libraries
      set_target_properties(${target_name} PROPERTIES MACOSX_RPATH ON)
    endif()
  endif()

  select_cxx_standard()
  if(NOT MSVC)
    # Make sure C++11/14 are enabled
    target_compile_options(${target_name} PUBLIC ${PYBIND11_CPP_STANDARD})
  endif()

  if(ARG_NO_EXTRAS)
    return()
  endif()

  # Find LTO flags
  _pybind11_find_lto_flags(lto_cxx_flags lto_linker_flags ARG_THIN_LTO)
  if (lto_cxx_flags OR lto_linker_flags)
    # Enable LTO flags if found, except for Debug builds
    separate_arguments(lto_cxx_flags)
    separate_arguments(lto_linker_flags)
    target_compile_options(${target_name} PRIVATE "$<$<NOT:$<CONFIG:Debug>>:${lto_cxx_flags}>")
    target_link_libraries(${target_name} PRIVATE "$<$<NOT:$<CONFIG:Debug>>:${lto_linker_flags}>")
  endif()

  # Set the default symbol visibility to hidden (very important to obtain small binaries)
  if (NOT MSVC)
    target_compile_options(${target_name} PRIVATE "-fvisibility=hidden")

    # Strip unnecessary sections of the binary on Linux/Mac OS
    if(CMAKE_STRIP)
      if(APPLE)
        add_custom_command(TARGET ${target_name} POST_BUILD
                           COMMAND ${CMAKE_STRIP} -x $<TARGET_FILE:${target_name}>)
      else()
        add_custom_command(TARGET ${target_name} POST_BUILD
                           COMMAND ${CMAKE_STRIP} $<TARGET_FILE:${target_name}>)
      endif()
    endif()
  endif()

  if(MSVC)
    # /MP enables multithreaded builds (relevant when there are many files), /bigobj is
    # needed for bigger binding projects due to the limit to 64k addressable sections
    target_compile_options(${target_name} PRIVATE /MP /bigobj)
  endif()
endfunction()
