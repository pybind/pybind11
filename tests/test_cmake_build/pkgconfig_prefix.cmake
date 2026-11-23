# Check that prefix= in the installed pybind11.pc resolves to the install prefix.
# Usage: cmake -DSOURCE_DIR=... -DWORK_DIR=... -DGENERATOR=... -DCXX_COMPILER=...
#        -P pkgconfig_prefix.cmake

foreach(datarootdir IN ITEMS "share" "." "./share" "lib/foo/share" "../shared-data")
  string(MAKE_C_IDENTIFIER "${datarootdir}" name)
  set(case_dir "${WORK_DIR}/${name}")
  set(build_dir "${case_dir}/build")
  set(install_dir "${case_dir}/install")
  file(REMOVE_RECURSE "${case_dir}")

  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -S "${SOURCE_DIR}" -B "${build_dir}" -G "${GENERATOR}"
      "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}" -DPYBIND11_TEST=OFF -DPYBIND11_NOPYTHON=ON
      "-DCMAKE_INSTALL_PREFIX=${install_dir}" "-DCMAKE_INSTALL_DATAROOTDIR=${datarootdir}"
    RESULT_VARIABLE result
    OUTPUT_QUIET)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Configure failed for CMAKE_INSTALL_DATAROOTDIR=${datarootdir}")
  endif()

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -P "${build_dir}/cmake_install.cmake"
    RESULT_VARIABLE result
    OUTPUT_QUIET)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Install failed for CMAKE_INSTALL_DATAROOTDIR=${datarootdir}")
  endif()

  set(pcfiledir "${install_dir}/${datarootdir}/pkgconfig")
  file(STRINGS "${pcfiledir}/pybind11.pc" prefix_line REGEX "^prefix=")
  string(REPLACE "prefix=" "" pc_prefix "${prefix_line}")
  string(REPLACE "\${pcfiledir}" "${pcfiledir}" pc_prefix "${pc_prefix}")
  get_filename_component(pc_prefix "${pc_prefix}" ABSOLUTE)
  get_filename_component(expected "${install_dir}" ABSOLUTE)

  if(NOT pc_prefix STREQUAL expected)
    message(FATAL_ERROR "CMAKE_INSTALL_DATAROOTDIR=${datarootdir}: ${prefix_line} resolves to "
                        "${pc_prefix}, expected ${expected}")
  endif()
  message(STATUS "CMAKE_INSTALL_DATAROOTDIR=${datarootdir}: ${prefix_line} OK")
endforeach()
