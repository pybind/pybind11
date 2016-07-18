# Set a default build configuration if none is specified. 'MinSizeRel' produces the smallest binaries
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'MinSizeRel' as none was specified.")
  set(CMAKE_BUILD_TYPE MinSizeRel CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

set(PYBIND11_EXAMPLES
  example-methods-and-attributes.cpp
  example-python-types.cpp
  example-operator-overloading.cpp
  example-constants-and-functions.cpp
  example-callbacks.cpp
  example-sequences-and-iterators.cpp
  example-buffers.cpp
  example-smart-ptr.cpp
  example-modules.cpp
  example-numpy-vectorize.cpp
  example-arg-keywords-and-defaults.cpp
  example-virtual-functions.cpp
  example-keep-alive.cpp
  example-opaque-types.cpp
  example-pickling.cpp
  example-inheritance.cpp
  example-stl-binder-vector.cpp
  example-eval.cpp
  example-custom-exceptions.cpp
  issues.cpp
)

# Check if Eigen is available
find_package(Eigen3 QUIET)

if(EIGEN3_FOUND)
  list(APPEND PYBIND11_EXAMPLES eigen.cpp)
  message(STATUS "Building Eigen testcase")
else()
  message(STATUS "NOT Building Eigen testcase")
endif()

# Create the binding library
pybind11_add_module(example example.cpp ${PYBIND11_EXAMPLES})
pybind11_enable_warnings(example)

if(EIGEN3_FOUND)
  target_include_directories(example PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_compile_definitions(example PRIVATE -DPYBIND11_TEST_EIGEN)
endif()

# Always write the output file directly into the 'example' directory (even on MSVC)
set(CompilerFlags
  LIBRARY_OUTPUT_DIRECTORY LIBRARY_OUTPUT_DIRECTORY_RELEASE LIBRARY_OUTPUT_DIRECTORY_DEBUG
  LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO
  RUNTIME_OUTPUT_DIRECTORY RUNTIME_OUTPUT_DIRECTORY_RELEASE RUNTIME_OUTPUT_DIRECTORY_DEBUG
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO)

foreach(CompilerFlag ${CompilerFlags})
  set_target_properties(example PROPERTIES ${CompilerFlag} ${PROJECT_SOURCE_DIR}/example)
endforeach()

set(RUN_TEST ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_test.py)
if(MSVC OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(RUN_TEST ${RUN_TEST} --relaxed)
endif()

foreach(VALUE ${PYBIND11_EXAMPLES})
  string(REGEX REPLACE "^(.+).cpp$" "\\1" EXAMPLE_NAME "${VALUE}")
  add_test(NAME ${EXAMPLE_NAME} COMMAND ${RUN_TEST} ${EXAMPLE_NAME})
endforeach()

