cmake_minimum_required(VERSION 3.7)

# The `cmake_minimum_required(VERSION 3.7...3.18)` syntax does not work with
# some versions of VS that have a patched CMake 3.11. This forces us to emulate
# the behavior using the following workaround:
if(${CMAKE_VERSION} VERSION_LESS 3.18)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else()
  cmake_policy(VERSION 3.18)
endif()

project(test_subdirectory_function CXX)

add_subdirectory("${PYBIND11_PROJECT_DIR}" pybind11)
pybind11_add_module(test_subdirectory_function THIN_LTO ../main.cpp)
set_target_properties(test_subdirectory_function PROPERTIES OUTPUT_NAME test_cmake_build)

add_custom_target(
  check_subdirectory_function
  ${CMAKE_COMMAND}
  -E
  env
  PYTHONPATH=$<TARGET_FILE_DIR:test_subdirectory_function>
  ${PYTHON_EXECUTABLE}
  ${PROJECT_SOURCE_DIR}/../test.py
  ${PROJECT_NAME})
