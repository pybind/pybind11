# Exercise target reuse and the actual link guard without rebuilding the precompiled library.
if(PYBIND11_FINDPYTHON
   AND NOT CMAKE_CROSSCOMPILING
   AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.26)
  find_package(Python REQUIRED COMPONENTS Interpreter)
endif()
if(PYBIND11_FINDPYTHON
   AND NOT CMAKE_CROSSCOMPILING
   AND Python_INTERPRETER_ID STREQUAL "Python"
   AND Python_VERSION VERSION_GREATER_EQUAL 3.13
   AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.26)
  set(_guard_options "-DPYBIND11_FINDPYTHON=ON" "-DPython_EXECUTABLE=${Python_EXECUTABLE}")
  if(DEFINED pybind11_SOURCE_DIR AND EXISTS "${pybind11_SOURCE_DIR}/CMakeLists.txt")
    list(APPEND _guard_options "-DGUARD_SOURCE_DIR=${pybind11_SOURCE_DIR}")
  else()
    list(APPEND _guard_options "-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
  endif()
  _pybind11_python_is_free_threaded(_guard_free_threaded)
  set(_guard_mismatches VERSION)
  if(NOT _guard_free_threaded)
    list(APPEND _guard_mismatches ABI3T)
  endif()
  foreach(_mismatch IN LISTS _guard_mismatches)
    execute_process(
      COMMAND
        "${CMAKE_COMMAND}" -S "${CMAKE_CURRENT_LIST_DIR}/precompile_guard" -B
        "${CMAKE_CURRENT_BINARY_DIR}/guard_${_mismatch}" ${_guard_options}
        "-DGUARD_MISMATCH=${_mismatch}"
      RESULT_VARIABLE _result
      OUTPUT_VARIABLE _out
      ERROR_VARIABLE _err)
    if(_result EQUAL 0 OR NOT "${_out}${_err}" MATCHES "same stable ABI target")
      message(FATAL_ERROR "Expected stable ABI ${_mismatch} mismatch rejection:\n${_out}${_err}")
    endif()
  endforeach()

  if(NOT _guard_free_threaded)
    # Keep strict executable linking on POSIX: Python::Module's macOS link options allow
    # undefined symbols and would hide the intentional guard mismatch.
    set(_guard_link_options)
    if(MSVC OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
      # Suppress automatic library names; the Windows module target supplies Python imports.
      # Python 3.13 does not yet provide Py_NO_LINK_LIB to disable the header pragmas.
      list(APPEND _guard_link_options LINKER:/NODEFAULTLIB:python3.lib
           "LINKER:/NODEFAULTLIB:python${Python_VERSION_MAJOR}${Python_VERSION_MINOR}_d.lib")
    endif()
    add_library(pybind11_guard_headers INTERFACE IMPORTED)
    set_target_properties(pybind11_guard_headers PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                                            "${pybind11_INCLUDE_DIRS}")
    set(_guard_link_libraries pybind11_guard_headers)
    if(WIN32)
      # Exported exception methods emit Python references even when the probe does not call them.
      list(APPEND _guard_link_libraries ${_Python}::Module)
    endif()
    foreach(_provider_api 0x030c0000 0x030d0000)
      set(_dir "${CMAKE_CURRENT_BINARY_DIR}/guard_link_${_provider_api}")
      file(MAKE_DIRECTORY "${_dir}")
      file(
        WRITE "${_dir}/provider.cpp"
        "#define PYBIND11_PRECOMPILED\n#define Py_LIMITED_API ${_provider_api}\n"
        "#include <pybind11/detail/internals.h>\n" "namespace pybind11 { namespace detail {\n"
        "void PYBIND11_PRECOMPILED_CONFIG_CHECK(precompiled_limited_api) {}\n}}\n")
      file(
        WRITE "${_dir}/consumer.cpp"
        "#define PYBIND11_PRECOMPILED\n#define Py_LIMITED_API 0x030C0000\n"
        "#include <pybind11/detail/internals.h>\n"
        "int main() { PYBIND11_PRECOMPILED_CONFIG_GUARD return 0; }\n")
      try_compile(
        _linked "${_dir}/build"
        SOURCES "${_dir}/provider.cpp" "${_dir}/consumer.cpp"
        LINK_OPTIONS ${_guard_link_options}
        LINK_LIBRARIES ${_guard_link_libraries} CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON
        OUTPUT_VARIABLE _out)
      if(_provider_api STREQUAL "0x030c0000")
        if(NOT _linked)
          message(FATAL_ERROR "Equivalent limited API versions did not link:\n${_out}")
        endif()
      elseif(_linked OR NOT "${_out}" MATCHES "pybind11_precompiled_config")
        message(FATAL_ERROR "Expected limited API link guard failure:\n${_out}")
      endif()
    endforeach()
  endif()
endif()
