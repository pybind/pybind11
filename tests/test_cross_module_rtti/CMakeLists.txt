possibly_uninitialized(PYTHON_MODULE_EXTENSION Python_INTERPRETER_ID)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)

if("${PYTHON_MODULE_EXTENSION}" MATCHES "pypy"
   OR "${Python_INTERPRETER_ID}" STREQUAL "PyPy"
   OR "${PYTHON_MODULE_EXTENSION}" MATCHES "graalpy")
  message(STATUS "Skipping visibility test on PyPy or GraalPy")
  add_custom_target(test_cross_module_rtti
  )# Dummy target on PyPy or GraalPy. Embedding is not supported.
  set(_suppress_unused_variable_warning "${DOWNLOAD_CATCH}")
  return()
endif()

if(TARGET Python::Module AND NOT TARGET Python::Python)
  message(STATUS "Skipping visibility test since no embed libs found")
  add_custom_target(test_cross_module_rtti) # Dummy target since embedding is not supported.
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

add_library(test_cross_module_rtti_lib SHARED lib.h lib.cpp)
add_library(test_cross_module_rtti_lib::test_cross_module_rtti_lib ALIAS
            test_cross_module_rtti_lib)
target_include_directories(test_cross_module_rtti_lib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(test_cross_module_rtti_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(test_cross_module_rtti_lib PUBLIC cxx_std_11)

generate_export_header(test_cross_module_rtti_lib)

pybind11_add_module(test_cross_module_rtti_bindings SHARED bindings.cpp)
target_link_libraries(test_cross_module_rtti_bindings
                      PUBLIC test_cross_module_rtti_lib::test_cross_module_rtti_lib)

add_executable(test_cross_module_rtti_main catch.cpp test_cross_module_rtti.cpp)
target_link_libraries(
  test_cross_module_rtti_main PUBLIC test_cross_module_rtti_lib::test_cross_module_rtti_lib
                                     pybind11::embed Catch2::Catch2)

# Ensure that we have built the python bindings since we load them in main
add_dependencies(test_cross_module_rtti_main test_cross_module_rtti_bindings)

pybind11_enable_warnings(test_cross_module_rtti_main)
pybind11_enable_warnings(test_cross_module_rtti_bindings)
pybind11_enable_warnings(test_cross_module_rtti_lib)

add_custom_target(
  test_cross_module_rtti
  COMMAND "$<TARGET_FILE:test_cross_module_rtti_main>"
  DEPENDS test_cross_module_rtti_main
  WORKING_DIRECTORY "$<TARGET_FILE_DIR:test_cross_module_rtti_main>")

set_target_properties(test_cross_module_rtti_bindings PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                                 "${CMAKE_CURRENT_BINARY_DIR}")

add_dependencies(check test_cross_module_rtti)
