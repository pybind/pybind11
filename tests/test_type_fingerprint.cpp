#include "pybind11_tests.h"

#define PYBIND11_PLATFORM_ABI_KEY                                                                 \
    PYBIND11_COMPILER_TYPE PYBIND11_STDLIB PYBIND11_BUILD_ABI PYBIND11_BUILD_TYPE

namespace {
namespace poc {

template <typename T>
std::string type_fingerprint() {
    return std::string(typeid(T).name()) + " " PYBIND11_PLATFORM_ABI_KEY;
}

} // namespace poc
} // namespace

TEST_SUBMODULE(type_fingerprint, m) { m.def("std_string", poc::type_fingerprint<std::string>); }
