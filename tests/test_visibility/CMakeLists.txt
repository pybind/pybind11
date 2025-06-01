possibly_uninitialized(PYTHON_MODULE_EXTENSION Python_INTERPRETER_ID)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

if("${PYTHON_MODULE_EXTENSION}" MATCHES "pypy"
   OR "${Python_INTERPRETER_ID}" STREQUAL "PyPy"
   OR "${PYTHON_MODULE_EXTENSION}" MATCHES "graalpy")
  message(STATUS "Skipping visibility test on PyPy or GraalPy")
  add_custom_target(test_visibility) # Dummy target on PyPy or GraalPy. Embedding is not supported.
  set(_suppress_unused_variable_warning "${DOWNLOAD_CATCH}")
  return()
endif()

if(TARGET Python::Module AND NOT TARGET Python::Python)
  message(STATUS "Skipping visibility test since no embed libs found")
  add_custom_target(test_visibility) # Dummy target since embedding is not supported.
  set(_suppress_unused_variable_warning "${DOWNLOAD_CATCH}")
  return()
endif()

find_package(Catch 2.13.10)

if(CATCH_FOUND)
  message(STATUS "Building interpreter tests using Catch v${CATCH_VERSION}")
else()
  message(STATUS "Catch not detected. Interpreter tests will be skipped. Install Catch headers"
                 " manually or use `cmake -DDOWNLOAD_CATCH=ON` to fetch them automatically.")
  return()
endif()

include(GenerateExportHeader)

add_library(test_visibility_lib SHARED lib.h lib.cpp)
add_library(test_visibility_lib::test_visibility_lib ALIAS test_visibility_lib)
target_include_directories(test_visibility_lib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(test_visibility_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(test_visibility_lib PUBLIC cxx_std_11)

generate_export_header(test_visibility_lib)

pybind11_add_module(test_visibility_bindings SHARED bindings.cpp)
target_link_libraries(test_visibility_bindings PUBLIC test_visibility_lib::test_visibility_lib)

add_executable(test_visibility_main catch.cpp test_visibility.cpp)
target_link_libraries(test_visibility_main PUBLIC test_visibility_lib::test_visibility_lib
                                                  pybind11::embed Catch2::Catch2)

# Ensure that we have built the python bindings since we load them in main
add_dependencies(test_visibility_main test_visibility_bindings)

pybind11_enable_warnings(test_visibility_main)
pybind11_enable_warnings(test_visibility_bindings)
pybind11_enable_warnings(test_visibility_lib)

add_custom_target(
  test_visibility
  COMMAND "$<TARGET_FILE:test_visibility_main>"
  DEPENDS test_visibility_main
  WORKING_DIRECTORY "$<TARGET_FILE_DIR:test_visibility_main>")

set_target_properties(test_visibility_bindings PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                          "${CMAKE_CURRENT_BINARY_DIR}")

add_dependencies(check test_visibility)
