cmake_minimum_required(VERSION 3.15...4.2)

project(test_subdirectory_embed CXX)

find_package(Python COMPONENTS Interpreter REQUIRED)

set(PYBIND11_INSTALL
    ON
    CACHE BOOL "")
set(PYBIND11_EXPORT_NAME test_export)

# Allow PYTHON_EXECUTABLE if in FINDPYTHON mode and building pybind11's tests
# (makes transition easier while we support both modes).
if(DEFINED PYTHON_EXECUTABLE AND NOT DEFINED Python_EXECUTABLE)
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

add_subdirectory("${pybind11_SOURCE_DIR}" pybind11)

find_package(Python REQUIRED COMPONENTS Development.Module OPTIONAL_COMPONENTS Development.Embed)
get_target_property(_pybind11_embed_links pybind11::embed INTERFACE_LINK_LIBRARIES)
if(TARGET Python::Python AND NOT "Python::Python" IN_LIST _pybind11_embed_links)
  message(FATAL_ERROR "pybind11::embed must link to Python::Python when Python::Python is available")
endif()

# Test basic target functionality
add_executable(test_subdirectory_embed ../embed.cpp)
target_link_libraries(test_subdirectory_embed PRIVATE pybind11::embed)
set_target_properties(test_subdirectory_embed PROPERTIES OUTPUT_NAME test_cmake_build)

add_custom_target(
  check_subdirectory_embed
  $<TARGET_FILE:test_subdirectory_embed> "${PROJECT_SOURCE_DIR}/../test.py"
  DEPENDS test_subdirectory_embed)

# Test custom export group -- PYBIND11_EXPORT_NAME
add_library(test_with_catch_lib ../embed.cpp)
target_link_libraries(test_with_catch_lib PRIVATE pybind11::embed)

install(
  TARGETS test_with_catch_lib
  EXPORT test_export
  ARCHIVE DESTINATION bin
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib)
install(EXPORT test_export DESTINATION lib/cmake/test_export/test_export-Targets.cmake)
