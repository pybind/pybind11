#include "pybind11_tests.h"

#include <memory>

namespace pybind11_tests {
namespace class_sh_trampoline_basic {

struct Abase {
    int val = 0;
    virtual ~Abase() = default;
    explicit Abase(int val_) : val{val_} {}
    int Get() const { return val * 10 + 3; }
    virtual int Add(int other_val) const = 0;

    // Some compilers complain about implicitly defined versions of some of the following:
    Abase(const Abase &) = default;
    Abase(Abase &&) noexcept = default;
    Abase &operator=(const Abase &) = default;
    Abase &operator=(Abase &&) noexcept = default;
};

struct AbaseAlias : Abase, py::trampoline_self_life_support {
    using Abase::Abase;

    int Add(int other_val) const override {
        PYBIND11_OVERRIDE_PURE(int,   /* Return type */
                               Abase, /* Parent class */
                               Add,   /* Name of function in C++ (must match Python name) */
                               other_val);
    }
};

int AddInCppRawPtr(const Abase *obj, int other_val) { return obj->Add(other_val) * 10 + 7; }

int AddInCppSharedPtr(const std::shared_ptr<Abase> &obj, int other_val) {
    return obj->Add(other_val) * 100 + 11;
}

int AddInCppUniquePtr(std::unique_ptr<Abase> obj, int other_val) {
    return obj->Add(other_val) * 100 + 13;
}

} // namespace class_sh_trampoline_basic
} // namespace pybind11_tests

using namespace pybind11_tests::class_sh_trampoline_basic;

TEST_SUBMODULE(class_sh_trampoline_basic, m) {
    py::classh<Abase, AbaseAlias>(m, "Abase")
        .def(py::init<int>(), py::arg("val"))
        .def("Get", &Abase::Get)
        .def("Add", &Abase::Add, py::arg("other_val"));

    m.def("AddInCppRawPtr", AddInCppRawPtr, py::arg("obj"), py::arg("other_val"));
    m.def("AddInCppSharedPtr", AddInCppSharedPtr, py::arg("obj"), py::arg("other_val"));
    m.def("AddInCppUniquePtr", AddInCppUniquePtr, py::arg("obj"), py::arg("other_val"));
}
