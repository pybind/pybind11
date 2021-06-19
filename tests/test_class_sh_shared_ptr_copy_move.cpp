#include "pybind11_tests.h"

#include <pybind11/smart_holder.h>

#include <memory>
#include <string>
#include <vector>

namespace pybind11_tests {
namespace class_sh_shared_ptr_copy_move {

template<int SerNo>
struct Foo {
    std::string mtxt;
    Foo() : mtxt("DefaultConstructor") {}
    Foo(const std::string &mtxt_) : mtxt(mtxt_) {}
    Foo(const Foo &other) { mtxt = other.mtxt + "_CpCtor"; }
    Foo(Foo &&other) { mtxt = other.mtxt + "_MvCtor"; }
};

using FooAVL = Foo<0>;
using FooDEF = Foo<1>;

} // namespace class_sh_shared_ptr_copy_move
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_shared_ptr_copy_move::FooAVL)

namespace pybind11_tests {
namespace class_sh_shared_ptr_copy_move {

TEST_SUBMODULE(class_sh_shared_ptr_copy_move, m) {
    namespace py = pybind11;

    py::class_<FooAVL, PYBIND11_SH_AVL(FooAVL)>(m, "FooAVL");
    py::class_<FooDEF, PYBIND11_SH_DEF(FooDEF)>(m, "FooDEF");

    m.def("test_avl_copy", []() {
        auto o = std::make_shared<FooAVL>("AVL");
        auto l = py::list();
        l.append(o);
    });
    m.def("test_def_copy", []() {
      auto o = std::make_shared<FooDEF>("DEF");
      auto l = py::list();
      l.append(o);
    });

    m.def("test_avl_move", []() {
      auto o = std::make_shared<FooAVL>("AVL");
      auto l = py::list();
      l.append(std::move(o));
    });
    m.def("test_def_move", []() {
      auto o = std::make_shared<FooDEF>("DEF");
      auto l = py::list();
      l.append(std::move(o));
    });
}

} // namespace class_sh_shared_ptr_copy_move
} // namespace pybind11_tests
