// The Catch implementation is compiled here. This is a standalone
// translation unit to avoid recompiling it for every test change.

#include <pybind11/embed.h>

// Silence MSVC C++17 deprecation warning from Catch regarding std::uncaught_exceptions (up to
// catch 2.0.1; this should be fixed in the next catch release after 2.0.1).
PYBIND11_WARNING_DISABLE_MSVC(4996)

// Catch uses _ internally, which breaks gettext style defines
#ifdef _
#    undef _
#endif

#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

int main(int argc, char *argv[]) {
    pybind11::scoped_interpreter guard{};
    auto result = Catch::Session().run(argc, argv);
    return result < 0xff ? result : 0xff;
}
