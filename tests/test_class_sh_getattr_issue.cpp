#include <pybind11/smart_holder.h>

#include "pybind11_tests.h"

namespace pybind11_tests {
namespace class_sh_getattr_issue {

// https://github.com/pybind/pybind11/issues/3788
struct Foo {
    Foo() = default;
    int bar() const { return 42; }
};

} // namespace class_sh_getattr_issue
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_getattr_issue::Foo)

TEST_SUBMODULE(class_sh_getattr_issue, m) {
    using namespace pybind11_tests::class_sh_getattr_issue;
    py::classh<Foo>(m, "Foo")
        .def(py::init<>())
        .def("bar", &Foo::bar)
        .def("__getattr__", [](Foo &, std::string key) { return "GetAttr: " + key; });
}
