#include "pybind11_tests.h"

#include <pybind11/smart_holder.h>

#include <memory>
#include <string>
#include <vector>

namespace pybind11_tests {
namespace {

const std::string fooNames[] = {"ShPtr_", "SmHld_"};

template <int SerNo>
struct Foo {
    std::string history;
    Foo(const std::string &history_) : history(history_) {}
    Foo(const Foo &other) : history(other.history + "_CpCtor") {}
    Foo(Foo &&other) : history(other.history + "_MvCtor") {}
    Foo &operator=(const Foo &other) {
        history = other.history + "_OpEqLv";
        return *this;
    }
    Foo &operator=(Foo &&other) {
        history = other.history + "_OpEqRv";
        return *this;
    }
    std::string get_history() const { return "Foo" + fooNames[SerNo] + history; }
};

using FooShPtr = Foo<0>;
using FooSmHld = Foo<1>;

} // namespace
} // namespace pybind11_tests

PYBIND11_TYPE_CASTER_BASE_HOLDER(pybind11_tests::FooShPtr, std::shared_ptr<pybind11_tests::FooShPtr>)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::FooSmHld)

namespace pybind11_tests {
namespace {

TEST_SUBMODULE(class_sh_shared_ptr_copy_move, m) {
    namespace py = pybind11;

    py::class_<FooShPtr, std::shared_ptr<FooShPtr>>(m, "FooShPtr")
        .def("get_history", &FooShPtr::get_history);
    py::classh<FooSmHld>(m, "FooSmHld")
        .def("get_history", &FooSmHld::get_history);

    m.def("test_ShPtr_copy", []() {
        auto o = std::make_shared<FooShPtr>("copy");
        auto l = py::list();
        l.append(o);
        return l;
    });
    m.def("test_SmHld_copy", []() {
        auto o = std::make_shared<FooSmHld>("copy");
        auto l = py::list();
        l.append(o);
        return l;
    });

    m.def("test_ShPtr_move", []() {
        auto o = std::make_shared<FooShPtr>("move");
        auto l = py::list();
        l.append(std::move(o));
        return l;
    });
    m.def("test_SmHld_move", []() {
        auto o = std::make_shared<FooSmHld>("move");
        auto l = py::list();
        l.append(std::move(o));
        return l;
    });
}

} // namespace
} // namespace pybind11_tests
