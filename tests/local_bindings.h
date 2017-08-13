#pragma once
#include "pybind11_tests.h"

/// Simple class used to test py::local:
template <int> class LocalBase {
public:
    LocalBase(int i) : i(i) { }
    int i = -1;
};

/// Registered with py::local in both main and secondary modules:
using LocalType = LocalBase<0>;
/// Registered without py::local in both modules:
using NonLocalType = LocalBase<1>;
/// A second non-local type (for stl_bind tests):
using NonLocal2 = LocalBase<2>;
/// Tests within-module, different-compilation-unit local definition conflict:
using LocalExternal = LocalBase<3>;
/// Mixed: registered local first, then global
using MixedLocalGlobal = LocalBase<4>;
/// Mixed: global first, then local (which fails)
using MixedGlobalLocal = LocalBase<5>;

using LocalVec = std::vector<LocalType>;
using LocalVec2 = std::vector<NonLocal2>;
using LocalMap = std::unordered_map<std::string, LocalType>;
using NonLocalVec = std::vector<NonLocalType>;
using NonLocalVec2 = std::vector<NonLocal2>;
using NonLocalMap = std::unordered_map<std::string, NonLocalType>;
using NonLocalMap2 = std::unordered_map<std::string, uint8_t>;

// Simple bindings (used with the above):
template <typename T, int Adjust, typename... Args>
py::class_<T> bind_local(Args && ...args) {
    return py::class_<T>(std::forward<Args>(args)...)
        .def(py::init<int>())
        .def("get", [](T &i) { return i.i + Adjust; });
};
