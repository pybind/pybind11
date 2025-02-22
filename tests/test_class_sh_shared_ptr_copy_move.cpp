#include "pybind11_tests.h"

#include <memory>
#include <string>
#include <vector>

namespace pybind11_tests {
namespace {

const std::string fooNames[] = {"ShPtr_", "SmHld_"};

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
using FooSmHld = Foo<1>;

struct Outer {
    std::shared_ptr<FooShPtr> ShPtr;
    std::shared_ptr<FooSmHld> SmHld;
    Outer()
        : ShPtr(std::make_shared<FooShPtr>("Outer")), SmHld(std::make_shared<FooSmHld>("Outer")) {}
    std::shared_ptr<FooShPtr> getShPtr() const { return ShPtr; }
    std::shared_ptr<FooSmHld> getSmHld() const { return SmHld; }
};

} // namespace

TEST_SUBMODULE(class_sh_shared_ptr_copy_move, m) {
    namespace py = pybind11;

    py::class_<FooShPtr, std::shared_ptr<FooShPtr>>(m, "FooShPtr")
        .def("get_history", &FooShPtr::get_history);
    py::classh<FooSmHld>(m, "FooSmHld").def("get_history", &FooSmHld::get_history);

    auto outer = py::class_<Outer>(m, "Outer").def(py::init());
#define MAKE_PROP(PropTyp)                                                                        \
    MAKE_PROP_FOO(ShPtr, PropTyp)                                                                 \
    MAKE_PROP_FOO(SmHld, PropTyp)

#define MAKE_PROP_FOO(FooTyp, PropTyp)                                                            \
    .def_##PropTyp(#FooTyp "_" #PropTyp "_default", &Outer::FooTyp)                               \
        .def_##PropTyp(                                                                           \
            #FooTyp "_" #PropTyp "_copy", &Outer::FooTyp, py::return_value_policy::copy)          \
        .def_##PropTyp(                                                                           \
            #FooTyp "_" #PropTyp "_move", &Outer::FooTyp, py::return_value_policy::move)
    outer MAKE_PROP(readonly) MAKE_PROP(readwrite);
#undef MAKE_PROP_FOO

#define MAKE_PROP_FOO(FooTyp, PropTyp)                                                            \
    .def_##PropTyp(#FooTyp "_property_" #PropTyp "_default", &Outer::FooTyp)                      \
        .def_property_##PropTyp(#FooTyp "_property_" #PropTyp "_copy",                            \
                                &Outer::get##FooTyp,                                              \
                                py::return_value_policy::copy)                                    \
        .def_property_##PropTyp(#FooTyp "_property_" #PropTyp "_move",                            \
                                &Outer::get##FooTyp,                                              \
                                py::return_value_policy::move)
    outer MAKE_PROP(readonly);
#undef MAKE_PROP_FOO
#undef MAKE_PROP

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

} // namespace pybind11_tests
