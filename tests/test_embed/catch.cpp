// The Catch implementation is compiled here. This is a standalone
// translation unit to avoid recompiling it for every test change.

#include <pybind11/embed.h>

#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

namespace py = pybind11;

int main(int argc, const char *argv[]) {
    py::scoped_interpreter guard{};
    auto result = Catch::Session().run(argc, argv);

    return result < 0xff ? result : 0xff;
}
