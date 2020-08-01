cmake_minimum_required(VERSION 3.7)

# The `cmake_minimum_required(VERSION 3.7...3.18)` syntax does not work with
# some versions of VS that have a patched CMake 3.11. This forces us to emulate
# the behavior using the following workaround:
if(${CMAKE_VERSION} VERSION_LESS 3.18)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.18)
endif()

project(test_installed_module CXX)

set(CMAKE_MODULE_PATH "")

find_package(pybind11 CONFIG REQUIRED)
message(
  STATUS "Found pybind11 v${pybind11_VERSION} ${pybind11_VERSION_TYPE}: ${pybind11_INCLUDE_DIRS}")

pybind11_add_module(test_cmake_build SHARED NO_EXTRAS ../main.cpp)

add_custom_target(
  check
  ${CMAKE_COMMAND}
  -E
  env
  PYTHONPATH=$<TARGET_FILE_DIR:test_cmake_build>
  ${PYTHON_EXECUTABLE}
  ${PROJECT_SOURCE_DIR}/../test.py
  ${PROJECT_NAME})
