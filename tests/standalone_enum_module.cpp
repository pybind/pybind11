// Copyright (c) 2026 The pybind Community.

#include <pybind11/pybind11.h>

namespace standalone_enum_module_ns {
enum SomeEnum {};
} // namespace standalone_enum_module_ns

using namespace standalone_enum_module_ns;

PYBIND11_MODULE(standalone_enum_module, m) {
    pybind11::enum_<SomeEnum> some_enum_wrapper(m, "SomeEnum");
}
