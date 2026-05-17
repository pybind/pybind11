cmake_minimum_required(VERSION 3.15...4.2)

project(test_installed_embed CXX)

if(PYBIND11_FINDPYTHON)
  find_package(Python COMPONENTS Interpreter REQUIRED)
endif()
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

if(PYBIND11_FINDPYTHON)
  if(CMAKE_VERSION VERSION_LESS 3.18)
    find_package(Python REQUIRED COMPONENTS Development)
  else()
    find_package(
      Python REQUIRED
      COMPONENTS Development.Module
      OPTIONAL_COMPONENTS Development.Embed)
  endif()

  get_target_property(_pybind11_embed_links pybind11::embed INTERFACE_LINK_LIBRARIES)
  if(TARGET Python::Python AND NOT "Python::Python" IN_LIST _pybind11_embed_links)
    message(
      FATAL_ERROR "pybind11::embed must link to Python::Python when Python::Python is available")
  endif()
endif()

add_executable(test_installed_embed ../embed.cpp)
target_link_libraries(test_installed_embed PRIVATE pybind11::embed)
set_target_properties(test_installed_embed PROPERTIES OUTPUT_NAME test_cmake_build)

# Do not treat includes from IMPORTED target as SYSTEM (Python headers in pybind11::embed).
# This may be needed to resolve header conflicts, e.g. between Python release and debug headers.
set_target_properties(test_installed_embed PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)

add_custom_target(
  check_installed_embed
  $<TARGET_FILE:test_installed_embed> ${PROJECT_SOURCE_DIR}/../test.py
  DEPENDS test_installed_embed)
