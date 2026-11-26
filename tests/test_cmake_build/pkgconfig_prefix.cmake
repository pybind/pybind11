# Check that prefix= in the installed pybind11.pc resolves to the install prefix.
# Usage: cmake -DSOURCE_DIR=... -DWORK_DIR=... -DGENERATOR=... -DCXX_COMPILER=...
#        [-DMAKE_PROGRAM=...] [-DGENERATOR_PLATFORM=...] [-DGENERATOR_TOOLSET=...]
#        [-DTOOLCHAIN_FILE=...] -P pkgconfig_prefix.cmake

set(extra_args "")
if(MAKE_PROGRAM)
  list(APPEND extra_args "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}")
endif()
if(GENERATOR_PLATFORM)
  list(APPEND extra_args -A "${GENERATOR_PLATFORM}")
endif()
if(GENERATOR_TOOLSET)
  list(APPEND extra_args -T "${GENERATOR_TOOLSET}")
endif()
if(TOOLCHAIN_FILE)
  list(APPEND extra_args "-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}")
endif()

# Check that prefix= resolves to the prefix and includedir= to the installed headers.
function(check_pc_file label pcfiledir expected_prefix)
  file(STRINGS "${pcfiledir}/pybind11.pc" prefix_line REGEX "^prefix=")
  string(REPLACE "prefix=" "" pc_prefix "${prefix_line}")
  string(REPLACE "\${pcfiledir}" "${pcfiledir}" pc_prefix "${pc_prefix}")
  get_filename_component(pc_prefix "${pc_prefix}" ABSOLUTE)
  get_filename_component(expected "${expected_prefix}" ABSOLUTE)
  if(NOT pc_prefix STREQUAL expected)
    message(FATAL_ERROR "${label}: ${prefix_line} resolves to ${pc_prefix}, expected ${expected}")
  endif()

  file(STRINGS "${pcfiledir}/pybind11.pc" includedir_line REGEX "^includedir=")
  string(REPLACE "includedir=" "" pc_includedir "${includedir_line}")
  string(REPLACE "\${prefix}" "${pc_prefix}" pc_includedir "${pc_includedir}")
  if(NOT EXISTS "${pc_includedir}/pybind11/pybind11.h")
    message(FATAL_ERROR "${label}: ${includedir_line} resolves to ${pc_includedir}, "
                        "which does not contain pybind11/pybind11.h")
  endif()
  message(STATUS "${label}: ${prefix_line} OK")
endfunction()

# Configure once (compiler detection is slow, especially on Windows), then
# reconfigure the same build directory for each case.
set(build_dir "${WORK_DIR}/build")
file(REMOVE_RECURSE "${WORK_DIR}")
set(configure_args -G "${GENERATOR}" ${extra_args} "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}"
                   -DPYBIND11_TEST=OFF -DPYBIND11_NOPYTHON=ON)

foreach(case IN ITEMS "share" "." "lib/foo/share" "../shared-data" "ABSOLUTE")
  string(MAKE_C_IDENTIFIER "${case}" name)
  set(install_dir "${WORK_DIR}/${name}/install")

  if(case STREQUAL "ABSOLUTE")
    set(datarootdir "${install_dir}/share")
    set(pcfiledir "${datarootdir}/pkgconfig")
  else()
    set(datarootdir "${case}")
    set(pcfiledir "${install_dir}/${datarootdir}/pkgconfig")
  endif()
  set(label "CMAKE_INSTALL_DATAROOTDIR=${datarootdir}")

  # :STRING stops old CMake versions from making a relative value absolute.
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -S "${SOURCE_DIR}" -B "${build_dir}" ${configure_args}
      "-DCMAKE_INSTALL_PREFIX=${install_dir}" "-DCMAKE_INSTALL_DATAROOTDIR:STRING=${datarootdir}"
    RESULT_VARIABLE result
    OUTPUT_QUIET)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Configure failed for ${label}")
  endif()
  set(configure_args "")

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -P "${build_dir}/cmake_install.cmake"
    RESULT_VARIABLE result
    OUTPUT_QUIET)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Install failed for ${label}")
  endif()
  check_pc_file("${label}" "${pcfiledir}" "${install_dir}")

  if(case STREQUAL "lib/foo/share")
    set(relocated_dir "${WORK_DIR}/${name}/relocated")
    execute_process(
      COMMAND "${CMAKE_COMMAND}" "-DCMAKE_INSTALL_PREFIX=${relocated_dir}" -P
              "${build_dir}/cmake_install.cmake"
      RESULT_VARIABLE result
      OUTPUT_QUIET)
    if(NOT result EQUAL 0)
      message(FATAL_ERROR "Relocated install failed for ${label}")
    endif()
    check_pc_file("${label} (relocated)" "${relocated_dir}/${datarootdir}/pkgconfig"
                  "${relocated_dir}")
  endif()
endforeach()
