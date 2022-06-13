cmake_minimum_required(VERSION 3.4)

# The `cmake_minimum_required(VERSION 3.4...3.18)` syntax does not work with
# some versions of VS that have a patched CMake 3.11. This forces us to emulate
# the behavior using the following workaround:
if(${CMAKE_VERSION} VERSION_LESS 3.18)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.18)
endif()

project(test_installed_static CXX)

find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

add_library(test_installed_static MODULE ../main.cpp)

target_link_libraries(test_installed_static PRIVATE pybind11::module pybind11::static)
set_target_properties(test_installed_static PROPERTIES OUTPUT_NAME test_cmake_build)

# Make sure result is, for example, test_installed_static.so, not libtest_installed_static.dylib
pybind11_extension(test_installed_static)

# Do not treat includes from IMPORTED target as SYSTEM (Python headers in pybind11::module).
# This may be needed to resolve header conflicts, e.g. between Python release and debug headers.
set_target_properties(test_installed_static PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)

if(DEFINED Python_EXECUTABLE)
  set(_Python_EXECUTABLE "${Python_EXECUTABLE}")
elseif(DEFINED PYTHON_EXECUTABLE)
  set(_Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
else()
  message(FATAL_ERROR "No Python executable defined (should not be possible at this stage)")
endif()

add_custom_target(
  check_installed_static
  ${CMAKE_COMMAND}
  -E
  env
  PYTHONPATH=$<TARGET_FILE_DIR:test_installed_static>
  ${_Python_EXECUTABLE}
  ${PROJECT_SOURCE_DIR}/../test.py
  ${PROJECT_NAME}
  DEPENDS test_installed_static)
