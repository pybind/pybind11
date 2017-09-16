if(${PYTHON_MODULE_EXTENSION} MATCHES "pypy")
  add_custom_target(cpptest)  # Dummy target on PyPy. Embedding is not supported.
  set(_suppress_unused_variable_warning "${DOWNLOAD_CATCH}")
  return()
endif()

find_package(Catch 1.9.3)
if(NOT CATCH_FOUND)
  message(STATUS "Catch not detected. Interpreter tests will be skipped. Install Catch headers"
                 " manually or use `cmake -DDOWNLOAD_CATCH=1` to fetch them automatically.")
  return()
endif()

add_executable(test_embed
  catch.cpp
  test_interpreter.cpp
)
target_include_directories(test_embed PRIVATE ${CATCH_INCLUDE_DIR})
pybind11_enable_warnings(test_embed)

if(NOT CMAKE_VERSION VERSION_LESS 3.0)
  target_link_libraries(test_embed PRIVATE pybind11::embed)
else()
  target_include_directories(test_embed PRIVATE ${PYBIND11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIRS})
  target_compile_options(test_embed PRIVATE ${PYBIND11_CPP_STANDARD})
  target_link_libraries(test_embed PRIVATE ${PYTHON_LIBRARIES})
endif()

find_package(Threads REQUIRED)
target_link_libraries(test_embed PUBLIC ${CMAKE_THREAD_LIBS_INIT})

if (WIN32)
    set(pathsep ";")
else()
    set(pathsep ":")
endif()

add_custom_target(cpptest COMMAND ${CMAKE_COMMAND} -E env
                    "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}${pathsep}${CMAKE_CURRENT_SOURCE_DIR}"
                    $<TARGET_FILE:test_embed>)

pybind11_add_module(external_module THIN_LTO external_module.cpp)
add_dependencies(cpptest external_module)

add_dependencies(check cpptest)
