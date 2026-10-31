# Optional OSS-Fuzz harness. Off by default so the normal test matrix does not
# need a libFuzzer toolchain. OSS-Fuzz and the corpus-replay CI job both build
# this same target.
option(PYBIND11_BUILD_FUZZERS "Build the OSS-Fuzz harnesses" OFF)

# Present even when the harness is not built, so a normal CI configure does not
# fail if something asks for the target by name.
function(pybind11_add_disabled_fuzz_replay)
  add_custom_target(
    fuzz_replay
    COMMAND ${CMAKE_COMMAND} -E echo "PYBIND11_BUILD_FUZZERS did not produce pybind_fuzzer"
    COMMAND ${CMAKE_COMMAND} -E false)
endfunction()

if(NOT PYBIND11_BUILD_FUZZERS)
  pybind11_add_disabled_fuzz_replay()
  return()
endif()

possibly_uninitialized(PYTHON_MODULE_EXTENSION Python_INTERPRETER_ID)

if("${PYTHON_MODULE_EXTENSION}" MATCHES "pypy"
   OR "${Python_INTERPRETER_ID}" STREQUAL "PyPy"
   OR "${PYTHON_MODULE_EXTENSION}" MATCHES "graalpy")
  message(STATUS "Skipping fuzz harness on PyPy or GraalPy")
  pybind11_add_disabled_fuzz_replay()
  return()
endif()

if(TARGET Python::Module AND NOT TARGET Python::Python)
  message(STATUS "Skipping fuzz harness since no embed libs found")
  pybind11_add_disabled_fuzz_replay()
  return()
endif()

set(_pybind11_fuzz_sources pybind_fuzzer.cc)
# OSS-Fuzz exports LIB_FUZZING_ENGINE to the object that provides main().
# Locally, link a small file-replay driver instead.
if(DEFINED ENV{LIB_FUZZING_ENGINE} AND NOT "$ENV{LIB_FUZZING_ENGINE}" STREQUAL "")
  set(_pybind11_fuzz_engine "$ENV{LIB_FUZZING_ENGINE}")
else()
  list(APPEND _pybind11_fuzz_sources fuzz_replay.cc)
  set(_pybind11_fuzz_engine "")
  set(_pybind11_fuzz_standalone ON)
endif()

add_executable(pybind_fuzzer ${_pybind11_fuzz_sources})
pybind11_enable_warnings(pybind_fuzzer)
target_link_libraries(pybind_fuzzer PRIVATE pybind11::embed)
if(_pybind11_fuzz_standalone)
  target_compile_definitions(pybind_fuzzer PRIVATE PYBIND11_FUZZER_STANDALONE)
else()
  target_link_libraries(pybind_fuzzer PRIVATE ${_pybind11_fuzz_engine})
endif()

file(GLOB _pybind11_fuzz_corpus CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/corpus/*")
add_custom_target(
  fuzz_replay
  COMMAND "$<TARGET_FILE:pybind_fuzzer>" ${_pybind11_fuzz_corpus}
  DEPENDS pybind_fuzzer
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  USES_TERMINAL
  COMMENT "Replaying the checked-in fuzz corpus")
