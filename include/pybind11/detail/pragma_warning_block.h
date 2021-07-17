#pragma once

#if defined(__INTEL_COMPILER)
#  pragma warning push
#  pragma warning disable 878   // incompatible exception specifications
#  pragma warning disable 2196  // warning #2196: routine is both "inline" and "noinline"
#elif defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4100) // warning C4100: Unreferenced formal parameter
#  pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#  pragma warning(disable: 4800) // warning C4800: 'int': forcing value to bool 'true' or 'false' (performance warning)
#  pragma warning(disable: 4996) // warning C4996: The POSIX name for this item is deprecated. Instead, use the ISO C and C++ conformant name
#  pragma warning(disable: 4522) // warning C4522: multiple assignment operators specified
#  pragma warning(disable: 4505) // warning C4505: 'PySlice_GetIndicesEx': unreferenced local function has been removed (PyPy only)
#elif defined(__GNUG__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#  pragma GCC diagnostic ignored "-Wattributes"
#  if __GNUC__ >= 7
#    pragma GCC diagnostic ignored "-Wnoexcept-type"
#  endif
#endif
