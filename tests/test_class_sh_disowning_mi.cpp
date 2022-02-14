#include <pybind11/smart_holder.h>

#include "pybind11_tests.h"

#include <memory>

namespace pybind11_tests {
namespace class_sh_disowning_mi {

// Diamond inheritance (copied from test_multiple_inheritance.cpp).
struct B {
    int val_b = 10;
    B() = default;
    B(const B &) = default;
    virtual ~B() = default;
};

struct C0 : public virtual B {
    int val_c0 = 20;
};

struct C1 : public virtual B {
    int val_c1 = 21;
};

struct D : public C0, public C1 {
    int val_d = 30;
};

void disown_b(std::unique_ptr<B>) {}

// test_multiple_inheritance_python
struct Base1 {
    explicit Base1(int i) : i(i) {}
    int foo() const { return i; }
    int i;
};

struct Base2 {
    explicit Base2(int j) : j(j) {}
    int bar() const { return j; }
    int j;
};

int disown_base1(std::unique_ptr<Base1> b1) { return b1->i * 2000 + 1; }
int disown_base2(std::unique_ptr<Base2> b2) { return b2->j * 2000 + 2; }

} // namespace class_sh_disowning_mi
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_disowning_mi::B)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_disowning_mi::C0)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_disowning_mi::C1)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_disowning_mi::D)

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_disowning_mi::Base1)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_disowning_mi::Base2)

TEST_SUBMODULE(class_sh_disowning_mi, m) {
    using namespace pybind11_tests::class_sh_disowning_mi;

    py::classh<B>(m, "B")
        .def(py::init<>())
        .def_readonly("val_b", &D::val_b)
        .def("b", [](B *self) { return self; })
        .def("get", [](const B &self) { return self.val_b; });

    py::classh<C0, B>(m, "C0")
        .def(py::init<>())
        .def_readonly("val_c0", &D::val_c0)
        .def("c0", [](C0 *self) { return self; })
        .def("get", [](const C0 &self) { return self.val_b * 100 + self.val_c0; });

    py::classh<C1, B>(m, "C1")
        .def(py::init<>())
        .def_readonly("val_c1", &D::val_c1)
        .def("c1", [](C1 *self) { return self; })
        .def("get", [](const C1 &self) { return self.val_b * 100 + self.val_c1; });

    py::classh<D, C0, C1>(m, "D")
        .def(py::init<>())
        .def_readonly("val_d", &D::val_d)
        .def("d", [](D *self) { return self; })
        .def("get", [](const D &self) {
            return self.val_b * 1000000 + self.val_c0 * 10000 + self.val_c1 * 100 + self.val_d;
        });

    m.def("disown_b", disown_b);

    // test_multiple_inheritance_python
    py::classh<Base1>(m, "Base1").def(py::init<int>()).def("foo", &Base1::foo);
    py::classh<Base2>(m, "Base2").def(py::init<int>()).def("bar", &Base2::bar);
    m.def("disown_base1", disown_base1);
    m.def("disown_base2", disown_base2);
}
