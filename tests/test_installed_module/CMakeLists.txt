cmake_minimum_required(VERSION 2.8.12)
project(test_installed_module CXX)

set(CMAKE_MODULE_PATH "")

find_package(pybind11 CONFIG REQUIRED)

message(STATUS "Found pybind11: ${pybind11_INCLUDE_DIRS} (found version ${pybind11_VERSION})")
message(STATUS "Found Python: ${PYTHON_INCLUDE_DIRS} (found version ${PYTHON_VERSION_STRING})")

pybind11_add_module(test_installed_module SHARED main.cpp)

add_custom_target(check ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:test_installed_module>
                  ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test.py)
