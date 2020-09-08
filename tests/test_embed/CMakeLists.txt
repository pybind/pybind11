if(${PYTHON_MODULE_EXTENSION} MATCHES "pypy")
  add_custom_target(cpptest) # Dummy target on PyPy. Embedding is not supported.
  set(_suppress_unused_variable_warning "${DOWNLOAD_CATCH}")
  return()
endif()

find_package(Catch 2.13.0)
if(CATCH_FOUND)
  message(STATUS "Building interpreter tests using Catch v${CATCH_VERSION}")
else()
  message(STATUS "Catch not detected. Interpreter tests will be skipped. Install Catch headers"
                 " manually or use `cmake -DDOWNLOAD_CATCH=ON` to fetch them automatically.")
  return()
endif()

add_executable(test_embed catch.cpp test_interpreter.cpp)
target_include_directories(test_embed PRIVATE "${CATCH_INCLUDE_DIR}")
pybind11_enable_warnings(test_embed)

target_link_libraries(test_embed PRIVATE pybind11::embed)

find_package(Threads REQUIRED)
target_link_libraries(test_embed PUBLIC Threads::Threads)

add_custom_target(
  cpptest
  COMMAND "$<TARGET_FILE:test_embed>"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

pybind11_add_module(external_module THIN_LTO external_module.cpp)
set_target_properties(external_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                                                 "${CMAKE_CURRENT_SOURCE_DIR}")
foreach(config ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${config} config)
  set_target_properties(external_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${config}
                                                   "${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()
add_dependencies(cpptest external_module)

add_dependencies(check cpptest)
