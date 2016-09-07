#pragma once
#include <pybind11/pybind11.h>
#include <functional>
#include <list>

namespace py = pybind11;
using namespace pybind11::literals;

class test_initializer {
public:
    test_initializer(std::function<void(py::module &)> initializer);
};
