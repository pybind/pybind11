// Copyright (c) 2026 The pybind Community.

#include <pybind11/pybind11.h>

namespace standalone_enum_module_ns {
enum class SomeEnum { value1, value2 };
}

using namespace standalone_enum_module_ns;

PYBIND11_MODULE(standalone_enum_module, m) {
    pybind11::enum_<SomeEnum>(m, "SomeEnum")
        .value("value1", SomeEnum::value1)
        .value("value2", SomeEnum::value2);
}
