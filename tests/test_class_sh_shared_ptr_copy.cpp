#include "pybind11_tests.h"

#include <pybind11/smart_holder.h>

#include <memory>
#include <string>
#include <vector>

namespace pybind11_tests {
namespace class_sh_shared_ptr_copy {

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

} // namespace class_sh_shared_ptr_copy
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_shared_ptr_copy::FooAVL)

namespace pybind11_tests {
namespace class_sh_shared_ptr_copy {

TEST_SUBMODULE(class_sh_shared_ptr_copy, m) {
    namespace py = pybind11;

    py::class_<FooAVL, PYBIND11_SH_AVL(FooAVL)>(m, "FooAVL");
    py::class_<FooDEF, std::shared_ptr<FooDEF>>(m, "FooDEF");

    m.def("test_avl", []() {
        auto o = std::make_shared<FooAVL>("AVL");
        auto l = py::list();
        l.append(o);
    });
    m.def("test_def", []() {
      auto o = std::make_shared<FooDEF>("DEF");
      auto l = py::list();
      l.append(o);
    });
}

} // namespace class_sh_shared_ptr_copy
} // namespace pybind11_tests
