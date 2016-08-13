if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting tests build type to MinSizeRel as none was specified")
  set(CMAKE_BUILD_TYPE MinSizeRel CACHE STRING "Choose the type of build" FORCE)
endif()

set(PYBIND11_TEST_FILES
  test_buffers.cpp
  test_callbacks.cpp
  test_constants_and_functions.cpp
  test_enum.cpp
  test_eval.cpp
  test_exceptions.cpp
  test_inheritance.cpp
  test_issues.cpp
  test_keep_alive.cpp
  test_kwargs_and_defaults.cpp
  test_methods_and_attributes.cpp
  test_modules.cpp
  test_numpy_dtypes.cpp
  test_numpy_vectorize.cpp
  test_opaque_types.cpp
  test_operator_overloading.cpp
  test_pickling.cpp
  test_python_types.cpp
  test_sequences_and_iterators.cpp
  test_smart_ptr.cpp
  test_stl_binders.cpp
  test_virtual_functions.cpp
)

# Check if Eigen is available
find_package(Eigen3 QUIET)

if(EIGEN3_FOUND)
  list(APPEND PYBIND11_TEST_FILES test_eigen.cpp)
  message(STATUS "Building tests with Eigen v${EIGEN3_VERSION}")
else()
  message(STATUS "Building tests WITHOUT Eigen")
endif()

# Create the binding library
pybind11_add_module(pybind11_tests pybind11_tests.cpp ${PYBIND11_TEST_FILES})
pybind11_enable_warnings(pybind11_tests)

if(EIGEN3_FOUND)
  target_include_directories(pybind11_tests PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_compile_definitions(pybind11_tests PRIVATE -DPYBIND11_TEST_EIGEN)
endif()

set(testdir ${PROJECT_SOURCE_DIR}/tests)

# Always write the output file directly into the 'tests' directory (even on MSVC)
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(pybind11_tests PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${testdir})
  foreach(config ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${config} config)
    set_target_properties(pybind11_tests PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${config} ${testdir})
  endforeach()
endif()

# Make sure pytest is found or try to install it if it's not found
macro(pybind11_execute_python)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -m ${ARGN} OUTPUT_QUIET ERROR_QUIET
                  RESULT_VARIABLE pybind11_execute_python_error)
endmacro()

if(NOT PYBIND11_PYTEST_FOUND)
  pybind11_execute_python(pytest --version --noconftest)
  if(pybind11_execute_python_error)
    message(STATUS "Installing pytest using pip")
    pybind11_execute_python(pip install pytest)
    if(pybind11_execute_python_error)
      message(STATUS "Installing pytest using pip --user (fallback)")
      pybind11_execute_python(pip install --user pytest)
    endif()

    pybind11_execute_python(pytest --version --noconftest)
    if(pybind11_execute_python_error)
      message(FATAL_ERROR "Running the tests requires pytest. Please install it manually.")
    endif()
  endif()

  set(PYBIND11_PYTEST_FOUND TRUE CACHE INTERAL "")
endif()

# A single command to compile and run the tests
add_custom_target(pytest COMMAND ${PYTHON_EXECUTABLE} -m pytest -rws
                  DEPENDS pybind11_tests WORKING_DIRECTORY ${testdir})
