#include <pybind11/smart_holder.h>

#include "pybind11_tests.h"

#include <memory>

namespace pybind11_tests {
namespace ccccccccccccccccccccc {

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

const std::string fooNames[] = {"ShPtr_"};

template <int SerNo>
struct Foo {
    std::string history;
    explicit Foo(const std::string &history_) : history(history_) {}
    Foo(const Foo &other) : history(other.history + "_CpCtor") {}
    Foo(Foo &&other) noexcept : history(other.history + "_MvCtor") {}
    Foo &operator=(const Foo &other) {
        history = other.history + "_OpEqLv";
        return *this;
    }
    Foo &operator=(Foo &&other) noexcept {
        history = other.history + "_OpEqRv";
        return *this;
    }
    std::string get_history() const { return "Foo" + fooNames[SerNo] + history; }
};

using FooShPtr = Foo<0>;

} // namespace ccccccccccccccccccccc
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::ccccccccccccccccccccc::B)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::ccccccccccccccccccccc::C0)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::ccccccccccccccccccccc::C1)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::ccccccccccccccccccccc::D)

PYBIND11_TYPE_CASTER_BASE_HOLDER(pybind11_tests::ccccccccccccccccccccc::FooShPtr,
                                 std::shared_ptr<pybind11_tests::ccccccccccccccccccccc::FooShPtr>)

TEST_SUBMODULE(ccccccccccccccccccccc, m) {
    using namespace pybind11_tests::ccccccccccccccccccccc;

    py::class_<FooShPtr, std::shared_ptr<FooShPtr>>(m, "FooShPtr")
        .def("get_history", &FooShPtr::get_history);

    m.def("test_ShPtr_copy", []() {
        auto o = std::make_shared<FooShPtr>("copy");
        auto l = py::list();
        l.append(o);
        return l;
    });

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
}
