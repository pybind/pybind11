#pragma once
#include <pybind11/pybind11.h>
#include <iostream>
#include <functional>
#include <list>

using std::cout;
using std::endl;

namespace py = pybind11;
using namespace pybind11::literals;

class test_initializer {
public:
    test_initializer(std::function<void(py::module &)> initializer);
};
