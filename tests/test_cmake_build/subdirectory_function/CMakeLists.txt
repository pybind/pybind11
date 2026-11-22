cmake_minimum_required(VERSION 3.15...4.2)

project(test_subdirectory_function CXX)

# Allow PYTHON_EXECUTABLE if in FINDPYTHON mode and building pybind11's tests
# (makes transition easier while we support both modes).
if(DEFINED PYTHON_EXECUTABLE AND NOT DEFINED Python_EXECUTABLE)
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()

add_subdirectory("${pybind11_SOURCE_DIR}" pybind11)
pybind11_add_module(test_subdirectory_function ../main.cpp)
set_target_properties(test_subdirectory_function PROPERTIES OUTPUT_NAME test_cmake_build)

# FindPython mode accepts WITH_SOABI, which CMake ignores for SHARED libraries.
if(COMMAND python_add_library OR COMMAND python3_add_library)
  pybind11_add_module(test_subdirectory_function_shared SHARED WITH_SOABI ../main.cpp)

  # An executable can link a SHARED module; it only links libpython itself on Apple.
  # Stable-ABI SHARED libraries never link libpython, and abi3t has no PyInit_ symbol.
  if(NOT WIN32
     AND NOT PYBIND11_STABLE_ABI
     AND (TARGET Python::Python OR TARGET Python3::Python))
    add_executable(test_subdirectory_function_exe ../link_shared.cpp)
    target_link_libraries(test_subdirectory_function_exe PRIVATE test_subdirectory_function_shared)
    if(APPLE)
      target_link_libraries(test_subdirectory_function_exe PRIVATE pybind11::embed)
    endif()
  endif()
endif()

if(DEFINED Python_EXECUTABLE)
  set(_Python_EXECUTABLE "${Python_EXECUTABLE}")
elseif(DEFINED PYTHON_EXECUTABLE)
  set(_Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
else()
  message(FATAL_ERROR "No Python executable defined (should not be possible at this stage)")
endif()

add_custom_target(
  check_subdirectory_function
  ${CMAKE_COMMAND}
  -E
  env
  PYTHONPATH=$<TARGET_FILE_DIR:test_subdirectory_function>
  ${_Python_EXECUTABLE}
  ${PROJECT_SOURCE_DIR}/../test.py
  ${PROJECT_NAME}
  DEPENDS test_subdirectory_function)

if(TARGET test_subdirectory_function_shared)
  add_dependencies(check_subdirectory_function test_subdirectory_function_shared)
endif()

if(TARGET test_subdirectory_function_exe)
  add_custom_command(
    TARGET check_subdirectory_function
    POST_BUILD
    COMMAND test_subdirectory_function_exe)
endif()
