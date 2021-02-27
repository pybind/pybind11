#include "pybind11_tests.h"

#include <pybind11/smart_holder.h>

#include <memory>

namespace pybind11_tests {
namespace test_class_sh_with_alias {

struct Abase {
    int val          = 0;
    virtual ~Abase() = default;
    Abase(int val_) : val{val_} {}
    int Get() const { return val * 10 + 3; }
    virtual int Add(int other_val) const = 0;

    // Some compilers complain about implicitly defined versions of some of the following:
    Abase(const Abase &) = default;
    Abase(Abase &&)      = default;
    Abase &operator=(const Abase &) = default;
    Abase &operator=(Abase &&) = default;
};

struct AbaseAlias : Abase {
    using Abase::Abase;

    int Add(int other_val) const override {
        PYBIND11_OVERRIDE_PURE(int,   /* Return type */
                               Abase, /* Parent class */
                               Add,   /* Name of function in C++ (must match Python name) */
                               other_val);
    }
};

int AddInCppRawPtr(const Abase *obj, int other_val) { return obj->Add(other_val) * 10 + 7; }

int AddInCppSharedPtr(std::shared_ptr<Abase> obj, int other_val) {
    return obj->Add(other_val) * 100 + 11;
}

int AddInCppUniquePtr(std::unique_ptr<Abase> obj, int other_val) {
    return obj->Add(other_val) * 100 + 13;
}

} // namespace test_class_sh_with_alias
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::test_class_sh_with_alias::Abase)

TEST_SUBMODULE(class_sh_with_alias, m) {
    using namespace pybind11_tests::test_class_sh_with_alias;

    py::classh<Abase, AbaseAlias>(m, "Abase")
        .def(py::init<int>(), py::arg("val"))
        .def("Get", &Abase::Get)
        .def("Add", &Abase::Add, py::arg("other_val"));

    m.def("AddInCppRawPtr", AddInCppRawPtr, py::arg("obj"), py::arg("other_val"));
    m.def("AddInCppSharedPtr", AddInCppSharedPtr, py::arg("obj"), py::arg("other_val"));
    m.def("AddInCppUniquePtr", AddInCppUniquePtr, py::arg("obj"), py::arg("other_val"));
}
