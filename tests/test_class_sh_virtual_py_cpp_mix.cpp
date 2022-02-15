#include <pybind11/smart_holder.h>

#include "pybind11_tests.h"

#include <memory>

namespace pybind11_tests {
namespace class_sh_virtual_py_cpp_mix {

class Base {
public:
    virtual ~Base() = default;
    virtual int get() const { return 101; }

    // Some compilers complain about implicitly defined versions of some of the following:
    Base() = default;
    Base(const Base &) = default;
};

class CppDerivedPlain : public Base {
public:
    int get() const override { return 202; }
};

class CppDerived : public Base {
public:
    int get() const override { return 212; }
};

int get_from_cpp_plainc_ptr(const Base *b) { return b->get() + 4000; }

int get_from_cpp_unique_ptr(std::unique_ptr<Base> b) { return b->get() + 5000; }

struct BaseVirtualOverrider : Base, py::trampoline_self_life_support {
    using Base::Base;

    int get() const override { PYBIND11_OVERRIDE(int, Base, get); }
};

struct CppDerivedVirtualOverrider : CppDerived, py::trampoline_self_life_support {
    using CppDerived::CppDerived;

    int get() const override { PYBIND11_OVERRIDE(int, CppDerived, get); }
};

} // namespace class_sh_virtual_py_cpp_mix
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_virtual_py_cpp_mix::Base)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_virtual_py_cpp_mix::CppDerivedPlain)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_virtual_py_cpp_mix::CppDerived)

TEST_SUBMODULE(class_sh_virtual_py_cpp_mix, m) {
    using namespace pybind11_tests::class_sh_virtual_py_cpp_mix;

    py::classh<Base, BaseVirtualOverrider>(m, "Base").def(py::init<>()).def("get", &Base::get);

    py::classh<CppDerivedPlain, Base>(m, "CppDerivedPlain").def(py::init<>());

    py::classh<CppDerived, Base, CppDerivedVirtualOverrider>(m, "CppDerived").def(py::init<>());

    m.def("get_from_cpp_plainc_ptr", get_from_cpp_plainc_ptr, py::arg("b"));
    m.def("get_from_cpp_unique_ptr", get_from_cpp_unique_ptr, py::arg("b"));
}
