cmake_minimum_required(VERSION 3.26...4.2)
project(test_precompile_guard CXX)

if(GUARD_SOURCE_DIR)
  add_subdirectory("${GUARD_SOURCE_DIR}" pybind11)
else()
  find_package(pybind11 CONFIG REQUIRED)
endif()

# Configuration only: no interpreter needs to support these target versions at run time.
set(PYBIND11_STABLE_ABI_VERSION 3.15)
set(PYBIND11_ABI3T OFF)
pybind11_precompile(STABLE_ABI)
if(GUARD_MISMATCH STREQUAL "VERSION")
  set(PYBIND11_STABLE_ABI_VERSION 3.16)
elseif(GUARD_MISMATCH STREQUAL "ABI3T")
  set(PYBIND11_ABI3T ON)
endif()
pybind11_precompile(STABLE_ABI)
