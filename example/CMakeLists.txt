include(CheckCXXCompilerFlag)

# Set a default build configuration if none is specified. 'MinSizeRel' produces the smallest binaries
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'MinSizeRel' as none was specified.")
  set(CMAKE_BUILD_TYPE MinSizeRel CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  # Enable link time optimization and set the default symbol
  # visibility to hidden (very important to obtain small binaries)
  if(NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
    # Check for Link Time Optimization support
    # (GCC/Clang)
    CHECK_CXX_COMPILER_FLAG("-flto" HAS_LTO_FLAG)
    if(HAS_LTO_FLAG)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
    endif()

    # Intel equivalent to LTO is called IPO
    if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
      CHECK_CXX_COMPILER_FLAG("-ipo" HAS_IPO_FLAG)
      if(HAS_IPO_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ipo")
      endif()
    endif()
  endif()
endif()

# Compile with compiler warnings turned on
if(MSVC)
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
endif()

set(PYBIND11_EXAMPLES
  example1.cpp
  example2.cpp
  example3.cpp
  example4.cpp
  example5.cpp
  example6.cpp
  example7.cpp
  example8.cpp
  example9.cpp
  example10.cpp
  example11.cpp
  example12.cpp
  example13.cpp
  example14.cpp
  example15.cpp
  example16.cpp
  example17.cpp
  issues.cpp
)

# Check if Eigen is available
find_package(Eigen3 QUIET)

if(EIGEN3_FOUND)
  list(APPEND PYBIND11_EXAMPLES eigen.cpp)
  message(STATUS "Building Eigen testcase")
else()
  message(STATUS "NOT Building Eigen testcase")
endif()

# Create the binding library
pybind11_add_module(example example.cpp ${PYBIND11_EXAMPLES})
if(EIGEN3_FOUND)
  target_include_directories(example PRIVATE ${EIGEN3_INCLUDE_DIR})
  target_compile_definitions(example PRIVATE -DPYBIND11_TEST_EIGEN)
endif()

# Always write the output file directly into the 'example' directory (even on MSVC)
set(CompilerFlags
  LIBRARY_OUTPUT_DIRECTORY LIBRARY_OUTPUT_DIRECTORY_RELEASE LIBRARY_OUTPUT_DIRECTORY_DEBUG
  LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO
  RUNTIME_OUTPUT_DIRECTORY RUNTIME_OUTPUT_DIRECTORY_RELEASE RUNTIME_OUTPUT_DIRECTORY_DEBUG
  RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO)

foreach(CompilerFlag ${CompilerFlags})
  set_target_properties(example PROPERTIES ${CompilerFlag} ${PROJECT_SOURCE_DIR}/example)
endforeach()

if(WIN32)
  if(MSVC)
    # /MP enables multithreaded builds (relevant when there are many files), /bigobj is
    # needed for bigger binding projects due to the limit to 64k addressable sections
    set_property(TARGET example APPEND PROPERTY COMPILE_OPTIONS /MP /bigobj)
    # Enforce size-based optimization and link time code generation on MSVC
    # (~30% smaller binaries in experiments); do nothing in debug mode.
    set_property(TARGET example APPEND PROPERTY COMPILE_OPTIONS
      "$<$<CONFIG:Release>:/Os>" "$<$<CONFIG:Release>:/GL>"
      "$<$<CONFIG:MinSizeRel>:/Os>" "$<$<CONFIG:MinSizeRel>:/GL>"
      "$<$<CONFIG:RelWithDebInfo>:/Os>" "$<$<CONFIG:RelWithDebInfo>:/GL>"
    )
    set_property(TARGET example APPEND_STRING PROPERTY LINK_FLAGS_RELEASE "/LTCG ")
    set_property(TARGET example APPEND_STRING PROPERTY LINK_FLAGS_MINSIZEREL "/LTCG ")
    set_property(TARGET example APPEND_STRING PROPERTY LINK_FLAGS_RELWITHDEBINFO "/LTCG ")
  endif()
elseif(UNIX)
  # Optimize for a small binary size
  if(NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
    set_target_properties(example PROPERTIES COMPILE_FLAGS "-Os")
  endif()

  # Strip unnecessary sections of the binary on Linux/Mac OS
  if(APPLE)
    if(NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
      add_custom_command(TARGET example POST_BUILD COMMAND strip -u -r $<TARGET_FILE:example>)
    endif()
  else()
    if(NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
      add_custom_command(TARGET example POST_BUILD COMMAND strip $<TARGET_FILE:example>)
    endif()
  endif()
endif()

set(RUN_TEST ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_test.py)
if(MSVC OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(RUN_TEST ${RUN_TEST} --relaxed)
endif()

foreach(VALUE ${PYBIND11_EXAMPLES})
  string(REGEX REPLACE "^(.+).cpp$" "\\1" EXAMPLE_NAME "${VALUE}")
  add_test(NAME ${EXAMPLE_NAME} COMMAND ${RUN_TEST} ${EXAMPLE_NAME})
endforeach()
