# CMakeLists.txt -- Build system for the pybind11 modules
#
# Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 3.7)

# VERSION 3.7...3.18, but some versions of MCVS have a patched CMake 3.11
# that do not work properly with this syntax, so using the following workaround:
if(${CMAKE_VERSION} VERSION_LESS 3.18)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.18)
endif()

# Extract project version from source
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/pybind11/detail/common.h"
     pybind11_version_defines REGEX "#define PYBIND11_VERSION_(MAJOR|MINOR|PATCH) ")

foreach(ver ${pybind11_version_defines})
  if(ver MATCHES [[#define PYBIND11_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$]])
    set(PYBIND11_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}")
  endif()
endforeach()

if(PYBIND11_VERSION_PATCH MATCHES [[([a-zA-Z]+)]])
  set(PYBIND11_VERSION_TYPE "${CMAKE_MATCH_1}")
endif()
string(REGEX MATCH "[0-9]+" PYBIND11_VERSION_PATCH "${PYBIND11_VERSION_PATCH}")

project(
  pybind11
  LANGUAGES CXX
  VERSION "${PYBIND11_VERSION_MAJOR}.${PYBIND11_VERSION_MINOR}.${PYBIND11_VERSION_PATCH}")

# Standard includes
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CMakeDependentOption)

message(STATUS "pybind11 v${pybind11_VERSION} ${PYBIND11_VERSION_TYPE}")

# Check if pybind11 is being used directly or via add_subdirectory
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(PYBIND11_MASTER_PROJECT ON)
  message(STATUS "CMake ${CMAKE_VERSION}")

  if(CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
  endif()
else()
  set(PYBIND11_MASTER_PROJECT OFF)
  set(pybind11_system SYSTEM)
endif()

option(PYBIND11_INSTALL "Install pybind11 header files?" ${PYBIND11_MASTER_PROJECT})
option(PYBIND11_TEST "Build pybind11 test suite?" ${PYBIND11_MASTER_PROJECT})
option(PYBIND11_CLASSIC_LTO "Use the classic LTO flag algorithm, even on CMake 3.9+" OFF)
cmake_dependent_option(
  USE_PYTHON_INCLUDE_DIR
  "Install pybind11 headers in Python include directory instead of default installation prefix"
  OFF "PYBIND11_INSTALL" OFF)

# NB: when adding a header don't forget to also add it to setup.py
set(PYBIND11_HEADERS
    include/pybind11/detail/class.h
    include/pybind11/detail/common.h
    include/pybind11/detail/descr.h
    include/pybind11/detail/init.h
    include/pybind11/detail/internals.h
    include/pybind11/detail/typeid.h
    include/pybind11/attr.h
    include/pybind11/buffer_info.h
    include/pybind11/cast.h
    include/pybind11/chrono.h
    include/pybind11/common.h
    include/pybind11/complex.h
    include/pybind11/options.h
    include/pybind11/eigen.h
    include/pybind11/embed.h
    include/pybind11/eval.h
    include/pybind11/iostream.h
    include/pybind11/functional.h
    include/pybind11/numpy.h
    include/pybind11/operators.h
    include/pybind11/pybind11.h
    include/pybind11/pytypes.h
    include/pybind11/stl.h
    include/pybind11/stl_bind.h)

# Compare with grep and warn if mismatched
if(PYBIND11_MASTER_PROJECT AND NOT CMAKE_VERSION VERSION_LESS 3.12)
  file(
    GLOB_RECURSE _pybind11_header_check
    LIST_DIRECTORIES false
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    CONFIGURE_DEPENDS "include/pybind11/*.h")
  set(_pybind11_here_only ${PYBIND11_HEADERS})
  set(_pybind11_disk_only ${_pybind11_header_check})
  list(REMOVE_ITEM _pybind11_here_only ${_pybind11_header_check})
  list(REMOVE_ITEM _pybind11_disk_only ${PYBIND11_HEADERS})
  if(_pybind11_here_only)
    message(AUTHOR_WARNING "PYBIND11_HEADERS has extra files:" ${_pybind11_here_only})
  endif()
  if(_pybind11_disk_only)
    message(AUTHOR_WARNING "PYBIND11_HEADERS is missing files:" ${_pybind11_disk_only})
  endif()
endif()

# CMake 3.12 added list(TRANSFORM <list> PREPEND
# But we can't use it yet
string(REPLACE "include/" "${CMAKE_CURRENT_SOURCE_DIR}/include/" PYBIND11_HEADERS
               "${PYBIND11_HEADERS}")

# Classic mode

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/tools")
include(pybind11Tools)

# Cache variables so pybind11_add_module can be used in parent projects
set(PYBIND11_INCLUDE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/include"
    CACHE INTERNAL "")
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
set(PYTHON_IS_DEBUG
    "${PYTHON_IS_DEBUG}"
    CACHE INTERNAL "")

if(PYBIND11_TEST OR (BUILD_TESTING AND PYBIND11_MASTER_PROJECT))
  add_subdirectory(tests)
endif()

if(USE_PYTHON_INCLUDE_DIR)
  file(RELATIVE_PATH CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX} ${PYTHON_INCLUDE_DIRS})
endif()

# Note: when creating targets, you cannot use if statements at configure time -
# you need generator expressions, because those will be placed in the target file.
# You can also place ifs *in* the Config.in, but not here.

# Build an interface library target:
add_library(pybind11 INTERFACE)
add_library(pybind11::pybind11 ALIAS pybind11) # to match exported target

target_include_directories(
  pybind11 ${pybind11_system} INTERFACE $<BUILD_INTERFACE:${PYBIND11_INCLUDE_DIR}>
                                        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# Only add Python for build - must be added during the import for config since it has to be re-discovered.
target_include_directories(pybind11 SYSTEM INTERFACE $<BUILD_INTERFACE:${PYTHON_INCLUDE_DIRS}>)

if(CMAKE_VERSION VERSION_LESS 3.13)
  target_compile_features(pybind11 INTERFACE cxx_inheriting_constructors cxx_user_literals
                                             cxx_right_angle_brackets)
else()
  # This was added in CMake 3.8, but we are keeping a consistent breaking
  # point for the config file at 3.13. A config generated by CMake 3.13+
  # can only be read in 3.13+ due to the SHELL usage later, so this is safe to do.
  target_compile_features(pybind11 INTERFACE cxx_std_11)
endif()

add_library(module INTERFACE)
add_library(pybind11::module ALIAS module)

target_link_libraries(module INTERFACE pybind11::pybind11)

# See https://github.com/Kitware/CMake/blob/master/Modules/CMakePlatformId.h.in for platform IDs
# Note: CMake 3.15 allows $<PLATFORM_ID:Windows,Cygwin>
target_link_libraries(
  module
  INTERFACE
    "$<$<OR:$<PLATFORM_ID:Windows>,$<PLATFORM_ID:Cygwin>>:$<BUILD_INTERFACE:${PYTHON_LIBRARIES}>>")

if(CMAKE_VERSION VERSION_LESS 3.13)
  target_link_libraries(module INTERFACE "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
else()
  # SHELL (3.12+) forces this to remain together, and link_options was added in 3.13+
  # This is safer, because you are ensured the deduplication pass in CMake will not consider
  # these separate and remove one but not the other.
  target_link_options(module INTERFACE "$<$<PLATFORM_ID:Darwin>:SHELL:-undefined dynamic_lookup>")
endif()

# Workaround for Python 2.7 and C++17 (C++14 as a warning) incompatibility
# This adds the flags -Wno-register and -Wno-deprecated-register if the compiler
# is Clang 3.9+ or AppleClang and the compile language is CXX, or /wd5033 for MSVC (all languages,
# since MSVC didn't recognize COMPILE_LANGUAGE until CMake 3.11+).
set(clang_4plus
    "$<AND:$<CXX_COMPILER_ID:Clang>,$<NOT:$<VERSION_LESS:$<CXX_COMPILER_VERSION>,3.9>>>")
set(no_register "$<OR:${clang_4plus},$<CXX_COMPILER_ID:AppleClang>>")
set(cxx_no_register "$<AND:$<COMPILE_LANGUAGE:CXX>,${no_register}>")
set(msvc "$<CXX_COMPILER_ID:MSVC>")
target_compile_options(
  pybind11 INTERFACE "$<${cxx_no_register}:-Wno-register;-Wno-deprecated-register>"
                     "$<${msvc}:/wd5033>")

add_library(embed INTERFACE)
add_library(pybind11::embed ALIAS embed)
target_link_libraries(embed INTERFACE pybind11::pybind11 $<BUILD_INTERFACE:${PYTHON_LIBRARIES}>)

if(PYBIND11_INSTALL)
  install(DIRECTORY ${PYBIND11_INCLUDE_DIR}/pybind11 DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  # GNUInstallDirs "DATADIR" wrong here; CMake search path wants "share".
  set(PYBIND11_CMAKECONFIG_INSTALL_DIR
      "share/cmake/${PROJECT_NAME}"
      CACHE STRING "install path for pybind11Config.cmake")

  configure_package_config_file(
    tools/${PROJECT_NAME}Config.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION ${PYBIND11_CMAKECONFIG_INSTALL_DIR})

  if(CMAKE_VERSION VERSION_LESS 3.14)
    # Remove CMAKE_SIZEOF_VOID_P from ConfigVersion.cmake since the library does
    # not depend on architecture specific settings or libraries.
    set(_PYBIND11_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
    unset(CMAKE_SIZEOF_VOID_P)

    write_basic_package_version_file(
      ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
      VERSION ${PROJECT_VERSION}
      COMPATIBILITY AnyNewerVersion)

    set(CMAKE_SIZEOF_VOID_P ${_PYBIND11_CMAKE_SIZEOF_VOID_P})
  else()
    # CMake 3.14+ natively supports header-only libraries
    write_basic_package_version_file(
      ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
      VERSION ${PROJECT_VERSION}
      COMPATIBILITY AnyNewerVersion ARCH_INDEPENDENT)
  endif()

  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
          tools/FindPythonLibsNew.cmake tools/pybind11Tools.cmake
    DESTINATION ${PYBIND11_CMAKECONFIG_INSTALL_DIR})

  if(NOT PYBIND11_EXPORT_NAME)
    set(PYBIND11_EXPORT_NAME "${PROJECT_NAME}Targets")
  endif()

  install(TARGETS pybind11 module embed EXPORT "${PYBIND11_EXPORT_NAME}")

  install(
    EXPORT "${PYBIND11_EXPORT_NAME}"
    NAMESPACE "pybind11::"
    DESTINATION ${PYBIND11_CMAKECONFIG_INSTALL_DIR})

  # Uninstall target
  if(PYBIND11_MASTER_PROJECT)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/tools/cmake_uninstall.cmake.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" IMMEDIATE @ONLY)

    add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P
                                        ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
  endif()
endif()
