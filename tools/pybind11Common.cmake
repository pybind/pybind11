#[======================================================[.rst

Adds the following targets::

    pybind11::pybind11 - link to headers and pybind11
    pybind11::module - Adds module links
    pybind11::embed - Adds embed links
    pybind11::lto - Link time optimizations (manual selection)
    pybind11::thin_lto - Link time optimizations (manual selection)
    pybind11::python_link_helper - Adds link to Python libraries
    pybind11::python2_no_register - Avoid warning/error with Python 2 + C++14/7
    pybind11::windows_extras - MSVC bigobj and mp for building multithreaded

Adds the following functions::

    pybind11_strip(target) - strip target after building on linux/macOS


#]======================================================]

# CMake 3.10 has an include_guard command, but we can't use that yet
if(TARGET pybind11::lto)
  return()
endif()

# If we are in subdirectory mode, all IMPORTED targets must be GLOBAL. If we
# are in CONFIG mode, they should be "normal" targets instead.
# In CMake 3.11+ you can promote a target to global after you create it,
# which might be simpler than this check.
get_property(
  is_config
  TARGET pybind11::headers
  PROPERTY IMPORTED)
if(NOT is_config)
  set(optional_global GLOBAL)
endif()

# --------------------- Shared targets ----------------------------

# Build an interface library target:
add_library(pybind11::pybind11 IMPORTED INTERFACE ${optional_global})
set_property(
  TARGET pybind11::pybind11
  APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES pybind11::headers)

# Build a module target:
add_library(pybind11::module IMPORTED INTERFACE ${optional_global})
set_property(
  TARGET pybind11::module
  APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES pybind11::pybind11)

# Build an embed library target:
add_library(pybind11::embed IMPORTED INTERFACE ${optional_global})
set_property(
  TARGET pybind11::embed
  APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES pybind11::pybind11)

# ----------------------- no register ----------------------

# Workaround for Python 2.7 and C++17 (C++14 as a warning) incompatibility
# This adds the flags -Wno-register and -Wno-deprecated-register if the compiler
# is Clang 3.9+ or AppleClang and the compile language is CXX, or /wd5033 for MSVC (all languages,
# since MSVC didn't recognize COMPILE_LANGUAGE until CMake 3.11+).

add_library(pybind11::python2_no_register INTERFACE IMPORTED ${optional_global})
set(clang_4plus
    "$<AND:$<CXX_COMPILER_ID:Clang>,$<NOT:$<VERSION_LESS:$<CXX_COMPILER_VERSION>,3.9>>>")
set(no_register "$<OR:${clang_4plus},$<CXX_COMPILER_ID:AppleClang>>")

if(MSVC AND CMAKE_VERSION VERSION_LESS 3.11)
  set(cxx_no_register "${no_register}")
else()
  set(cxx_no_register "$<AND:$<COMPILE_LANGUAGE:CXX>,${no_register}>")
endif()

set(msvc "$<CXX_COMPILER_ID:MSVC>")

set_property(
  TARGET pybind11::python2_no_register
  PROPERTY INTERFACE_COMPILE_OPTIONS
           "$<${cxx_no_register}:-Wno-register;-Wno-deprecated-register>" "$<${msvc}:/wd5033>")

# --------------------------- link helper ---------------------------

add_library(pybind11::python_link_helper IMPORTED INTERFACE ${optional_global})

if(CMAKE_VERSION VERSION_LESS 3.13)
  # In CMake 3.11+, you can set INTERFACE properties via the normal methods, and
  # this would be simpler.
  set_property(
    TARGET pybind11::python_link_helper
    APPEND
    PROPERTY INTERFACE_LINK_LIBRARIES "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
else()
  # link_options was added in 3.13+
  # This is safer, because you are ensured the deduplication pass in CMake will not consider
  # these separate and remove one but not the other.
  set_property(
    TARGET pybind11::python_link_helper
    APPEND
    PROPERTY INTERFACE_LINK_OPTIONS "$<$<PLATFORM_ID:Darwin>:LINKER:-undefined,dynamic_lookup>")
endif()

# ------------------------ Windows extras -------------------------

add_library(pybind11::windows_extras IMPORTED INTERFACE ${optional_global})

if(MSVC)
  # /MP enables multithreaded builds (relevant when there are many files), /bigobj is
  # needed for bigger binding projects due to the limit to 64k addressable sections
  set_property(
    TARGET pybind11::windows_extras
    APPEND
    PROPERTY INTERFACE_COMPILE_OPTIONS /bigobj)

  if(CMAKE_VERSION VERSION_LESS 3.11)
    set_property(
      TARGET pybind11::windows_extras
      APPEND
      PROPERTY INTERFACE_COMPILE_OPTIONS $<$<NOT:$<CONFIG:Debug>>:/MP>)
  else()
    # Only set these options for C++ files.  This is important so that, for
    # instance, projects that include other types of source files like CUDA
    # .cu files don't get these options propagated to nvcc since that would
    # cause the build to fail.
    set_property(
      TARGET pybind11::windows_extras
      APPEND
      PROPERTY INTERFACE_COMPILE_OPTIONS $<$<NOT:$<CONFIG:Debug>>:$<$<COMPILE_LANGUAGE:CXX>:/MP>>)
  endif()
endif()

# ----------------------- Legacy option --------------------------

# Warn or error if old variable name used
if(PYBIND11_CPP_STANDARD)
  string(REGEX MATCH [[..$]] VAL "${PYBIND11_CPP_STANDARD}")
  if(CMAKE_CXX_STANDARD)
    if(NOT CMAKE_CXX_STANDARD STREQUAL VAL)
      message(WARNING "CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} does not match "
                      "PYBIND11_CPP_STANDARD=${PYBIND11_CPP_STANDARD}, "
                      "please remove PYBIND11_CPP_STANDARD from your cache")
    endif()
  else()
    set(supported_standards 11 14 17 20)
    if("${VAL}" IN_LIST supported_standards)
      message(WARNING "USE -DCMAKE_CXX_STANDARD=${VAL} instead of PYBIND11_PYTHON_VERSION")
      set(CMAKE_CXX_STANDARD
          ${VAL}
          CACHE STRING "From PYBIND11_CPP_STANDARD")
    else()
      message(FATAL_ERROR "PYBIND11_CPP_STANDARD should be replaced with CMAKE_CXX_STANDARD "
                          "(last two chars: ${VAL} not understood as a valid CXX std)")
    endif()
  endif()
endif()

# --------------------- Python specifics -------------------------

# Check to see which Python mode we are in, new, old, or no python
if(PYBIND11_NOPYTHON)
  set(_pybind11_nopython ON)
elseif(
  PYBIND11_FINDPYTHON
  OR Python_FOUND
  OR Python2_FOUND
  OR Python3_FOUND)
  # New mode
  include("${CMAKE_CURRENT_LIST_DIR}/pybind11NewTools.cmake")

else()

  # Classic mode
  include("${CMAKE_CURRENT_LIST_DIR}/pybind11Tools.cmake")

endif()

# --------------------- LTO -------------------------------

include(CheckCXXCompilerFlag)

# Checks whether the given CXX/linker flags can compile and link a cxx file.
# cxxflags and linkerflags are lists of flags to use.  The result variable is a
# unique variable name for each set of flags: the compilation result will be
# cached base on the result variable.  If the flags work, sets them in
# cxxflags_out/linkerflags_out internal cache variables (in addition to
# ${result}).
function(_pybind11_return_if_cxx_and_linker_flags_work result cxxflags linkerflags cxxflags_out
         linkerflags_out)
  set(CMAKE_REQUIRED_LIBRARIES ${linkerflags})
  check_cxx_compiler_flag("${cxxflags}" ${result})
  if(${result})
    set(${cxxflags_out}
        "${cxxflags}"
        PARENT_SCOPE)
    set(${linkerflags_out}
        "${linkerflags}"
        PARENT_SCOPE)
  endif()
endfunction()

function(_pybind11_generate_lto target prefer_thin_lto)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(cxx_append "")
    set(linker_append "")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT APPLE)
      # Clang Gold plugin does not support -Os; append -O3 to MinSizeRel builds to override it
      set(linker_append ";$<$<CONFIG:MinSizeRel>:-O3>")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
      set(cxx_append ";-fno-fat-lto-objects")
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND prefer_thin_lto)
      _pybind11_return_if_cxx_and_linker_flags_work(
        HAS_FLTO_THIN "-flto=thin${cxx_append}" "-flto=thin${linker_append}"
        PYBIND11_LTO_CXX_FLAGS PYBIND11_LTO_LINKER_FLAGS)
    endif()

    if(NOT HAS_FLTO_THIN)
      _pybind11_return_if_cxx_and_linker_flags_work(
        HAS_FLTO "-flto${cxx_append}" "-flto${linker_append}" PYBIND11_LTO_CXX_FLAGS
        PYBIND11_LTO_LINKER_FLAGS)
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    # Intel equivalent to LTO is called IPO
    _pybind11_return_if_cxx_and_linker_flags_work(HAS_INTEL_IPO "-ipo" "-ipo"
                                                  PYBIND11_LTO_CXX_FLAGS PYBIND11_LTO_LINKER_FLAGS)
  elseif(MSVC)
    # cmake only interprets libraries as linker flags when they start with a - (otherwise it
    # converts /LTCG to \LTCG as if it was a Windows path).  Luckily MSVC supports passing flags
    # with - instead of /, even if it is a bit non-standard:
    _pybind11_return_if_cxx_and_linker_flags_work(HAS_MSVC_GL_LTCG "/GL" "-LTCG"
                                                  PYBIND11_LTO_CXX_FLAGS PYBIND11_LTO_LINKER_FLAGS)
  endif()

  # Enable LTO flags if found, except for Debug builds
  if(PYBIND11_LTO_CXX_FLAGS)
    set(not_debug "$<NOT:$<CONFIG:Debug>>")
    set(cxx_lang "$<COMPILE_LANGUAGE:CXX>")
    if(MSVC AND CMAKE_VERSION VERSION_LESS 3.11)
      set(genex "${not_debug}")
    else()
      set(genex "$<AND:${not_debug},${cxx_lang}>")
    endif()
    set_property(
      TARGET ${target}
      APPEND
      PROPERTY INTERFACE_COMPILE_OPTIONS "$<${genex}:${PYBIND11_LTO_CXX_FLAGS}>")
    if(CMAKE_PROJECT_NAME STREQUAL "pybind11")
      message(STATUS "${target} enabled")
    endif()
  else()
    if(CMAKE_PROJECT_NAME STREQUAL "pybind11")
      message(STATUS "${target} disabled (not supported by the compiler and/or linker)")
    endif()
  endif()

  if(PYBIND11_LTO_LINKER_FLAGS)
    if(CMAKE_VERSION VERSION_LESS 3.11)
      set_property(
        TARGET ${target}
        APPEND
        PROPERTY INTERFACE_LINK_LIBRARIES "$<${not_debug}:${PYBIND11_LTO_LINKER_FLAGS}>")
    else()
      set_property(
        TARGET ${target}
        APPEND
        PROPERTY INTERFACE_LINK_OPTIONS "$<${not_debug}:${PYBIND11_LTO_LINKER_FLAGS}>")
    endif()
  endif()
endfunction()

add_library(pybind11::lto IMPORTED INTERFACE ${optional_global})
_pybind11_generate_lto(pybind11::lto FALSE)

add_library(pybind11::thin_lto IMPORTED INTERFACE ${optional_global})
_pybind11_generate_lto(pybind11::thin_lto TRUE)

# ---------------------- pybind11_strip -----------------------------

function(pybind11_strip target_name)
  # Strip unnecessary sections of the binary on Linux/Mac OS
  if(CMAKE_STRIP)
    if(APPLE)
      set(x_opt -x)
    endif()

    add_custom_command(
      TARGET ${target_name}
      POST_BUILD
      COMMAND ${CMAKE_STRIP} ${x_opt} $<TARGET_FILE:${target_name}>)
  endif()
endfunction()
