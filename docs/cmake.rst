.. _cmake:

Building with CMake
===================

The following snippet should be a good starting point to create bindings across
platforms. It assumes that the code is located in a file named :file:`example.cpp`,
and that the pybind11 repository is located in a subdirectory named :file:`pybind11`.

.. code-block:: cmake

    cmake_minimum_required(VERSION 2.8)

    project(example)

    # Add a CMake parameter for choosing a desired Python version
    set(EXAMPLE_PYTHON_VERSION "" CACHE STRING "Python version to use for compiling the example library")

    # Set a default build configuration if none is specified. 'MinSizeRel' produces the smallest binaries
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
      message(STATUS "Setting build type to 'MinSizeRel' as none was specified.")
      set(CMAKE_BUILD_TYPE MinSizeRel CACHE STRING "Choose the type of build." FORCE)
      set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
    endif()
    string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

    # Try to autodetect Python (can be overridden manually if needed)
    set(Python_ADDITIONAL_VERSIONS 3.4 3.5 3.6)
    find_package(PythonLibs ${EXAMPLE_PYTHON_VERSION} REQUIRED)

    if (UNIX)
      # Enable C++11 mode
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

      # Enable link time optimization and set the default symbol
      # visibility to hidden (very important to obtain small binaries)
      if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -flto")
      endif()
    endif()

    # Include path for Python header files
    include_directories(${PYTHON_INCLUDE_DIR})

    # Include path for pybind11 header files -- this may need to be changed depending on your setup
    include_directories(${PROJECT_SOURCE_DIR}/pybind11/include)

    # Create the binding library
    add_library(example SHARED
      example.cpp
      # ... extra files go here ...
    )

    # Don't add a 'lib' prefix to the shared library
    set_target_properties(example PROPERTIES PREFIX "")

    if (WIN32)
      if (MSVC)
        # Enforce size-based optimization and link time code generation
        # on MSVC (~30% smaller binaries in experiments). /bigobj is needed
        # for bigger binding projects due to the limit to 64k addressable sections
        # /MP enables multithreaded builds (relevant when there are many files).
        set_target_properties(example PROPERTIES COMPILE_FLAGS "/Os /GL /MP /bigobj")
        set_target_properties(example PROPERTIES LINK_FLAGS "/LTCG")
      endif()

      # .PYD file extension on Windows
      set_target_properties(example PROPERTIES SUFFIX ".pyd")

      # Link against the Python shared library
      target_link_libraries(example ${PYTHON_LIBRARY})
    elseif (UNIX)
      # It's quite common to have multiple copies of the same Python version
      # installed on one's system. E.g.: one copy from the OS and another copy
      # that's statically linked into an application like Blender or Maya.
      # If we link our plugin library against the OS Python here and import it
      # into Blender or Maya later on, this will cause segfaults when multiple
      # conflicting Python instances are active at the same time.

      # Windows is not affected by this issue since it handles DLL imports
      # differently. The solution for Linux and Mac OS is simple: we just don't
      # link against the Python library. The resulting shared library will have
      # missing symbols, but that's perfectly fine -- they will be resolved at
      # import time.

      # .SO file extension on Linux/Mac OS
      set_target_properties(example PROPERTIES SUFFIX ".so")

      # Strip unnecessary sections of the binary on Linux/Mac OS
      if(APPLE)
        set_target_properties(example PROPERTIES MACOSX_RPATH ".")
        set_target_properties(example PROPERTIES LINK_FLAGS "-undefined dynamic_lookup -dead_strip")
        if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
          add_custom_command(TARGET example POST_BUILD COMMAND strip -u -r ${PROJECT_BINARY_DIR}/example.so)
        endif()
      else()
        if (NOT ${U_CMAKE_BUILD_TYPE} MATCHES DEBUG)
          add_custom_command(TARGET example POST_BUILD COMMAND strip ${PROJECT_BINARY_DIR}/example.so)
        endif()
      endif()
    endif()
