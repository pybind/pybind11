cmake_minimum_required(VERSION 3.0)
project(test_installed_target CXX)

set(CMAKE_MODULE_PATH "")

find_package(pybind11 CONFIG REQUIRED)

message(STATUS "Found pybind11: ${pybind11_INCLUDE_DIRS} (found version ${pybind11_VERSION})")
message(STATUS "Found Python: ${PYTHON_INCLUDE_DIRS} (found version ${PYTHON_VERSION_STRING})")

add_library(test_installed_target MODULE main.cpp)

target_link_libraries(test_installed_target PRIVATE pybind11::pybind11)

# make sure result is, for example, test_installed_target.so, not libtest_installed_target.dylib
set_target_properties(test_installed_target PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                                       SUFFIX "${PYTHON_MODULE_EXTENSION}")

add_custom_target(check ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:test_installed_target>
                  ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test.py)
