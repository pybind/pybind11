# CMakeLists.txt -- Build system for the pybind11 modules
#
# Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

cmake_minimum_required(VERSION 2.8.12)

if (POLICY CMP0048)
  # cmake warns if loaded from a min-3.0-required parent dir, so silence the warning:
  cmake_policy(SET CMP0048 NEW)
endif()

project(pybind11)

# Check if pybind11 is being used directly or via add_subdirectory
set(PYBIND11_MASTER_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(PYBIND11_MASTER_PROJECT ON)
endif()

option(PYBIND11_INSTALL "Install pybind11 header files?" ${PYBIND11_MASTER_PROJECT})
option(PYBIND11_TEST    "Build pybind11 test suite?"     ${PYBIND11_MASTER_PROJECT})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/tools")

include(pybind11Tools)

# Cache variables so pybind11_add_module can be used in parent projects
set(PYBIND11_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/include" CACHE INTERNAL "")
set(PYTHON_INCLUDE_DIRS ${PYTHON_INCLUDE_DIRS} CACHE INTERNAL "")
set(PYTHON_LIBRARIES ${PYTHON_LIBRARIES} CACHE INTERNAL "")
set(PYTHON_MODULE_PREFIX ${PYTHON_MODULE_PREFIX} CACHE INTERNAL "")
set(PYTHON_MODULE_EXTENSION ${PYTHON_MODULE_EXTENSION} CACHE INTERNAL "")

set(PYBIND11_HEADERS
  include/pybind11/attr.h
  include/pybind11/cast.h
  include/pybind11/chrono.h
  include/pybind11/class_support.h
  include/pybind11/common.h
  include/pybind11/complex.h
  include/pybind11/descr.h
  include/pybind11/options.h
  include/pybind11/eigen.h
  include/pybind11/eval.h
  include/pybind11/functional.h
  include/pybind11/numpy.h
  include/pybind11/operators.h
  include/pybind11/pybind11.h
  include/pybind11/pytypes.h
  include/pybind11/stl.h
  include/pybind11/stl_bind.h
  include/pybind11/typeid.h
)
string(REPLACE "include/" "${CMAKE_CURRENT_SOURCE_DIR}/include/"
       PYBIND11_HEADERS "${PYBIND11_HEADERS}")

if (PYBIND11_TEST)
  add_subdirectory(tests)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# extract project version from source
file(STRINGS "${PYBIND11_INCLUDE_DIR}/pybind11/common.h" pybind11_version_defines
     REGEX "#define PYBIND11_VERSION_(MAJOR|MINOR|PATCH) ")
foreach(ver ${pybind11_version_defines})
  if (ver MATCHES "#define PYBIND11_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
    set(PYBIND11_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
  endif()
endforeach()
set(${PROJECT_NAME}_VERSION ${PYBIND11_VERSION_MAJOR}.${PYBIND11_VERSION_MINOR}.${PYBIND11_VERSION_PATCH})
message(STATUS "pybind11 v${${PROJECT_NAME}_VERSION}")

option (USE_PYTHON_INCLUDE_DIR "Install pybind11 headers in Python include directory instead of default installation prefix" OFF)
if (USE_PYTHON_INCLUDE_DIR)
    file(RELATIVE_PATH CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX} ${PYTHON_INCLUDE_DIRS})
endif()

if(NOT (CMAKE_VERSION VERSION_LESS 3.0))  # CMake >= 3.0
  # Build an interface library target:
  add_library(module INTERFACE)
  target_include_directories(module INTERFACE $<BUILD_INTERFACE:${PYBIND11_INCLUDE_DIR}>
                                              $<BUILD_INTERFACE:${PYTHON_INCLUDE_DIRS}>
                                              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  if(WIN32 OR CYGWIN)
    target_link_libraries(module INTERFACE $<BUILD_INTERFACE:${PYTHON_LIBRARIES}>)
  elseif(APPLE)
    target_link_libraries(module INTERFACE "-undefined dynamic_lookup")
  endif()
  target_compile_options(module INTERFACE $<BUILD_INTERFACE:${PYBIND11_CPP_STANDARD}>)

  add_library(pybind11::module ALIAS module)  # to match exported target
endif()

if (PYBIND11_INSTALL)
  install(FILES ${PYBIND11_HEADERS}
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/pybind11)
  # GNUInstallDirs "DATADIR" wrong here; CMake search path wants "share".
  set(PYBIND11_CMAKECONFIG_INSTALL_DIR "share/cmake/${PROJECT_NAME}" CACHE STRING "install path for pybind11Config.cmake")

  configure_package_config_file(tools/${PROJECT_NAME}Config.cmake.in
                                "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
                                INSTALL_DESTINATION ${PYBIND11_CMAKECONFIG_INSTALL_DIR})
  write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                                   VERSION ${${PROJECT_NAME}_VERSION}
                                   COMPATIBILITY AnyNewerVersion)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
                tools/FindPythonLibsNew.cmake
                tools/pybind11Tools.cmake
          DESTINATION ${PYBIND11_CMAKECONFIG_INSTALL_DIR})

  if(NOT (CMAKE_VERSION VERSION_LESS 3.0))
    install(TARGETS module
            EXPORT "${PROJECT_NAME}Targets")
    install(EXPORT "${PROJECT_NAME}Targets"
            NAMESPACE "${PROJECT_NAME}::"
            DESTINATION ${PYBIND11_CMAKECONFIG_INSTALL_DIR})
  endif()
endif()
