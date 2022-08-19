// The Catch implementation is compiled here. This is a standalone
// translation unit to avoid recompiling it for every test change.

#include <pybind11/embed.h>

#ifdef _MSC_VER
// Silence MSVC C++17 deprecation warning from Catch regarding std::uncaught_exceptions (up to
// catch 2.0.1; this should be fixed in the next catch release after 2.0.1).
#    pragma warning(disable : 4996)
#endif

// Catch uses _ internally, which breaks gettext style defines
#ifdef _
#    undef _
#endif

#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

namespace py = pybind11;

int main(int argc, char *argv[]) {
    // Setup for TEST_CASE in test_interpreter.cpp, tagging on a large random number:
    std::string updated_pythonpath("/pybind11_test_embed_PYTHONPATH_2099743835476552");
    const char *preexisting_pythonpath = getenv("PYTHONPATH");
    if (preexisting_pythonpath != nullptr) {
#if defined(_MSC_VER) || defined(__MINGW32__)
        updated_pythonpath += ';';
#else
        updated_pythonpath += ':';
#endif
        updated_pythonpath += preexisting_pythonpath;
    }
    setenv("PYTHONPATH", updated_pythonpath.c_str(), /*overwrite=*/1);

    py::scoped_interpreter guard{};

    auto result = Catch::Session().run(argc, argv);

    return result < 0xff ? result : 0xff;
}
