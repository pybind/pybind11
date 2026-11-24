if(NOT PYBIND11_BUILD_FUZZERS)
  return()
endif()

possibly_uninitialized(PYTHON_MODULE_EXTENSION Python_INTERPRETER_ID)

if("${PYTHON_MODULE_EXTENSION}" MATCHES "pypy"
   OR "${Python_INTERPRETER_ID}" STREQUAL "PyPy"
   OR "${PYTHON_MODULE_EXTENSION}" MATCHES "graalpy")
  message(STATUS "Skipping fuzz harness on PyPy or GraalPy")
  return()
endif()

if(TARGET Python::Module AND NOT TARGET Python::Python)
  message(STATUS "Skipping fuzz harness since no embed libs found")
  return()
endif()

if(NOT PYBIND11_FUZZER_INCLUDE_DIR)
  find_program(_pybind11_fuzz_clang NAMES clang)
  if(_pybind11_fuzz_clang)
    execute_process(
      COMMAND "${_pybind11_fuzz_clang}" -print-resource-dir
      OUTPUT_VARIABLE _pybind11_fuzz_resource_dir
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
endif()
find_path(
  PYBIND11_FUZZER_INCLUDE_DIR fuzzer/FuzzedDataProvider.h
  HINTS "${_pybind11_fuzz_resource_dir}/include"
  DOC "Directory containing fuzzer/FuzzedDataProvider.h")
if(NOT EXISTS "${PYBIND11_FUZZER_INCLUDE_DIR}/fuzzer/FuzzedDataProvider.h")
  message(
    FATAL_ERROR
      "FuzzedDataProvider.h not found. Install Clang's development headers or set "
      "PYBIND11_FUZZER_INCLUDE_DIR to the directory containing fuzzer/FuzzedDataProvider.h")
endif()

# Copy only this portable header: Clang's full resource include directory also
# contains compiler-specific standard headers that cannot be used with GCC.
configure_file("${PYBIND11_FUZZER_INCLUDE_DIR}/fuzzer/FuzzedDataProvider.h"
               "${CMAKE_CURRENT_BINARY_DIR}/include/fuzzer/FuzzedDataProvider.h" COPYONLY)

set(_pybind11_fuzz_sources pybind_fuzzer.cpp)
# OSS-Fuzz exports LIB_FUZZING_ENGINE to the object that provides main().
# Locally, link a small file-replay driver instead.
if(DEFINED ENV{LIB_FUZZING_ENGINE} AND NOT "$ENV{LIB_FUZZING_ENGINE}" STREQUAL "")
  set(_pybind11_fuzz_engine "$ENV{LIB_FUZZING_ENGINE}")
else()
  list(APPEND _pybind11_fuzz_sources fuzz_replay.cpp)
  set(_pybind11_fuzz_engine "")
endif()

add_executable(pybind_fuzzer ${_pybind11_fuzz_sources})
add_executable(test_fuzz_input test_fuzz_input.cpp)
foreach(target pybind_fuzzer test_fuzz_input)
  pybind11_enable_warnings(${target})
  target_include_directories(${target} SYSTEM PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/include")
endforeach()
target_link_libraries(pybind_fuzzer PRIVATE pybind11::embed)
if(_pybind11_fuzz_engine)
  target_link_libraries(pybind_fuzzer PRIVATE ${_pybind11_fuzz_engine})
endif()

file(GLOB _pybind11_fuzz_corpus CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/corpus/*")
add_custom_target(
  fuzz_replay
  COMMAND "$<TARGET_FILE:test_fuzz_input>" "${CMAKE_CURRENT_SOURCE_DIR}/corpus/exec_success"
  COMMAND "$<TARGET_FILE:pybind_fuzzer>" ${_pybind11_fuzz_corpus}
  DEPENDS pybind_fuzzer test_fuzz_input
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  USES_TERMINAL
  COMMENT "Replaying the checked-in fuzz corpus")
