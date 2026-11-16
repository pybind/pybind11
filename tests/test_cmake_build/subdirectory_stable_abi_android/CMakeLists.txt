cmake_minimum_required(VERSION 3.26...4.2)

project(test_subdirectory_stable_abi_android CXX)

# Host simulation of the Android link workaround, not a native Android build.
set(ANDROID TRUE)
set(PYBIND11_FINDPYTHON NEW)
set(Python_ARTIFACTS_INTERACTIVE TRUE)

add_subdirectory("${pybind11_SOURCE_DIR}" pybind11)
pybind11_add_module(test_cmake_build STABLE_ABI ../main.cpp)
target_link_options(test_cmake_build PRIVATE "LINKER:--no-undefined")

add_custom_target(
  check_subdirectory_stable_abi_android
  ${CMAKE_COMMAND}
  -E
  env
  PYTHONPATH=$<TARGET_FILE_DIR:test_cmake_build>
  ${Python_EXECUTABLE}
  ${PROJECT_SOURCE_DIR}/../test.py
  ${PROJECT_NAME}
  DEPENDS test_cmake_build)
