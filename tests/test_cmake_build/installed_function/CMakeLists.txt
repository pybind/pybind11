cmake_minimum_required(VERSION 3.15...3.30)

project(test_installed_function CXX)

find_package(pybind11 CONFIG REQUIRED)
message(
  STATUS "Found pybind11 v${pybind11_VERSION} ${pybind11_VERSION_TYPE}: ${pybind11_INCLUDE_DIRS}")

pybind11_add_module(test_installed_function SHARED NO_EXTRAS ../main.cpp)
set_target_properties(test_installed_function PROPERTIES OUTPUT_NAME test_cmake_build)

if(DEFINED Python_EXECUTABLE)
  set(_Python_EXECUTABLE "${Python_EXECUTABLE}")
elseif(DEFINED PYTHON_EXECUTABLE)
  set(_Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
else()
  message(FATAL_ERROR "No Python executable defined (should not be possible at this stage)")
endif()

add_custom_target(
  check_installed_function
  ${CMAKE_COMMAND}
  -E
  env
  PYTHONPATH=$<TARGET_FILE_DIR:test_installed_function>
  ${_Python_EXECUTABLE}
  ${PROJECT_SOURCE_DIR}/../test.py
  ${PROJECT_NAME}
  DEPENDS test_installed_function)
