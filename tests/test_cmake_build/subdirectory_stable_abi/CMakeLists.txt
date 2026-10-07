cmake_minimum_required(VERSION 3.26...4.2)

project(test_subdirectory_stable_abi CXX)

# Allow PYTHON_EXECUTABLE if in FINDPYTHON mode and building pybind11's tests
# (makes transition easier while we support both modes).
if(DEFINED PYTHON_EXECUTABLE AND NOT DEFINED Python_EXECUTABLE)
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

# Keep FindPython in pybind11's directory scope; the results must still reach us.
set(Python_ARTIFACTS_INTERACTIVE TRUE)
set(PYBIND11_FINDPYTHON NEW)

add_subdirectory("${pybind11_SOURCE_DIR}" pybind11)
pybind11_add_module(test_subdirectory_stable_abi STABLE_ABI ../main.cpp)
set_target_properties(test_subdirectory_stable_abi PROPERTIES OUTPUT_NAME test_cmake_build)

if(DEFINED Python_EXECUTABLE)
  set(_Python_EXECUTABLE "${Python_EXECUTABLE}")
elseif(DEFINED PYTHON_EXECUTABLE)
  set(_Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
else()
  message(FATAL_ERROR "No Python executable defined (should not be possible at this stage)")
endif()

add_custom_target(
  check_subdirectory_stable_abi
  ${CMAKE_COMMAND}
  -E
  env
  PYTHONPATH=$<TARGET_FILE_DIR:test_subdirectory_stable_abi>
  PYBIND11_EXPECT_ABI3=$<TARGET_FILE_NAME:test_subdirectory_stable_abi>
  ${_Python_EXECUTABLE}
  ${PROJECT_SOURCE_DIR}/../test.py
  ${PROJECT_NAME}
  DEPENDS test_subdirectory_stable_abi)
