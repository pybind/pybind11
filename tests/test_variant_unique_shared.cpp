#include "pybind11_tests.h"

#include <pybind11/vptr_holder.h>

#include <memory>
#include <variant>

namespace pybind11_tests {

using pybind11::vptr;

vptr<double> from_raw() { return vptr<double>{new double{3}}; }

vptr<double> from_unique() {
    return vptr<double>{std::unique_ptr<double>(new double{5})};
}

vptr<double> from_shared() {
    return vptr<double>{std::shared_ptr<double>(new double{7})};
}

TEST_SUBMODULE(variant_unique_shared, m) {

    m.def("from_raw", from_raw);
    m.def("from_unique", from_unique);
    m.def("from_shared", from_shared);

    py::class_<vptr<double>>(m, "vptr_double")
        .def(py::init<>())
        .def("ownership_type", &vptr<double>::ownership_type)
        .def("get_value",
             [](vptr<double> &v) {
                 auto p = v.get();
                 if (p)
                     return *p;
                 return -1.;
             })
        .def("get_unique",
             [](vptr<double> &v) {
                 v.get_unique();
                 return;
             })
        .def("get_shared", [](vptr<double> &v) {
            v.get_shared();
            return;
        });
}

} // namespace pybind11_tests
