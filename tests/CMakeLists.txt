if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting tests build type to MinSizeRel as none was specified")
  set(CMAKE_BUILD_TYPE MinSizeRel CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# Full set of test files (you can override these; see below)
set(PYBIND11_TEST_FILES
  test_alias_initialization.cpp
  test_buffers.cpp
  test_callbacks.cpp
  test_chrono.cpp
  test_class_args.cpp
  test_constants_and_functions.cpp
  test_copy_move_policies.cpp
  test_docstring_options.cpp
  test_eigen.cpp
  test_enum.cpp
  test_eval.cpp
  test_exceptions.cpp
  test_inheritance.cpp
  test_issues.cpp
  test_keep_alive.cpp
  test_kwargs_and_defaults.cpp
  test_methods_and_attributes.cpp
  test_modules.cpp
  test_multiple_inheritance.cpp
  test_numpy_array.cpp
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

# Invoking cmake with something like:
#     cmake -DPYBIND11_TEST_OVERRIDE="test_issues.cpp;test_picking.cpp" ..
# lets you override the tests that get compiled and run.  You can restore to all tests with:
#     cmake -DPYBIND11_TEST_OVERRIDE= ..
if (PYBIND11_TEST_OVERRIDE)
  set(PYBIND11_TEST_FILES ${PYBIND11_TEST_OVERRIDE})
endif()

string(REPLACE ".cpp" ".py" PYBIND11_PYTEST_FILES "${PYBIND11_TEST_FILES}")

# Check if Eigen is available; if not, remove from PYBIND11_TEST_FILES (but
# keep it in PYBIND11_PYTEST_FILES, so that we get the "eigen is not installed"
# skip message).
list(FIND PYBIND11_TEST_FILES test_eigen.cpp PYBIND11_TEST_FILES_EIGEN_I)
if(PYBIND11_TEST_FILES_EIGEN_I GREATER -1)
  find_package(Eigen3 QUIET)

  if(EIGEN3_FOUND)
    message(STATUS "Building tests with Eigen v${EIGEN3_VERSION}")
  else()
    list(REMOVE_AT PYBIND11_TEST_FILES ${PYBIND11_TEST_FILES_EIGEN_I})
    message(STATUS "Building tests WITHOUT Eigen")
  endif()
endif()

# Create the binding library
pybind11_add_module(pybind11_tests pybind11_tests.cpp
  ${PYBIND11_TEST_FILES} ${PYBIND11_HEADERS})

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

# Make sure pytest is found or produce a fatal error
if(NOT PYBIND11_PYTEST_FOUND)
  execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pytest --version --noconftest OUTPUT_QUIET ERROR_QUIET
                  RESULT_VARIABLE PYBIND11_EXEC_PYTHON_ERR)
  if(PYBIND11_EXEC_PYTHON_ERR)
    message(FATAL_ERROR "Running the tests requires pytest.  Please install it manually (try: ${PYTHON_EXECUTABLE} -m pip install pytest)")
  endif()
  set(PYBIND11_PYTEST_FOUND TRUE CACHE INTERNAL "")
endif()

# A single command to compile and run the tests
add_custom_target(pytest COMMAND ${PYTHON_EXECUTABLE} -m pytest -rws ${PYBIND11_PYTEST_FILES}
                  DEPENDS pybind11_tests WORKING_DIRECTORY ${testdir})

if(PYBIND11_TEST_OVERRIDE)
  add_custom_command(TARGET pytest POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Note: not all tests run: -DPYBIND11_TEST_OVERRIDE is in effect")
endif()

# And another to show the .so size and, if a previous size, compare it:
add_custom_command(TARGET pybind11_tests POST_BUILD
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/libsize.py
  $<TARGET_FILE:pybind11_tests> ${CMAKE_CURRENT_BINARY_DIR}/sosize-$<TARGET_FILE_NAME:pybind11_tests>.txt)

# Test CMake build using functions and targets from subdirectory or installed location
add_custom_target(test_cmake_build)
if(NOT CMAKE_VERSION VERSION_LESS 3.1)
  # 3.0 needed for interface library for subdirectory_target/installed_target
  # 3.1 needed for cmake -E env for testing

  include(CMakeParseArguments)
  function(pybind11_add_build_test name)
    cmake_parse_arguments(ARG "INSTALL" "" "" ${ARGN})

    set(build_options "-DCMAKE_PREFIX_PATH=${PROJECT_BINARY_DIR}/mock_install"
                      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                      "-DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}"
                      "-DPYBIND11_CPP_STANDARD=${PYBIND11_CPP_STANDARD}")
    if(NOT ARG_INSTALL)
      list(APPEND build_options "-DPYBIND11_PROJECT_DIR=${PROJECT_SOURCE_DIR}")
    endif()

    add_custom_target(test_${name} ${CMAKE_CTEST_COMMAND}
      --quiet --output-log test_cmake_build/${name}.log
      --build-and-test "${CMAKE_CURRENT_SOURCE_DIR}/test_cmake_build/${name}"
                       "${CMAKE_CURRENT_BINARY_DIR}/test_cmake_build/${name}"
      --build-config Release
      --build-noclean
      --build-generator ${CMAKE_GENERATOR}
      $<$<BOOL:${CMAKE_GENERATOR_PLATFORM}>:--build-generator-platform> ${CMAKE_GENERATOR_PLATFORM}
      --build-makeprogram ${CMAKE_MAKE_PROGRAM}
      --build-target check
      --build-options ${build_options}
    )
    if(ARG_INSTALL)
      add_dependencies(test_${name} mock_install)
    endif()
    add_dependencies(test_cmake_build test_${name})
  endfunction()

  pybind11_add_build_test(subdirectory_function)
  pybind11_add_build_test(subdirectory_target)

  if(PYBIND11_INSTALL)
    add_custom_target(mock_install ${CMAKE_COMMAND}
      "-DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/mock_install"
      -P "${PROJECT_BINARY_DIR}/cmake_install.cmake"
    )

    pybind11_add_build_test(installed_function INSTALL)
    pybind11_add_build_test(installed_target INSTALL)
  endif()
endif()

# Run all the tests
add_custom_target(check DEPENDS pytest test_cmake_build)
