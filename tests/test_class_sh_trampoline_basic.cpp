#include <pybind11/smart_holder.h>

#include "pybind11_tests.h"

#include <memory>

namespace pybind11_tests {
namespace class_sh_trampoline_basic {

template <int SerNo> // Using int as a trick to easily generate a series of types.
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

template <int SerNo>
struct AbaseAlias : Abase<SerNo> {
    using Abase<SerNo>::Abase;

    int Add(int other_val) const override {
        PYBIND11_OVERRIDE_PURE(int,          /* Return type */
                               Abase<SerNo>, /* Parent class */
                               Add,          /* Name of function in C++ (must match Python name) */
                               other_val);
    }
};

template <>
struct AbaseAlias<1> : Abase<1>, py::trampoline_self_life_support {
    using Abase<1>::Abase;

    int Add(int other_val) const override {
        PYBIND11_OVERRIDE_PURE(int,      /* Return type */
                               Abase<1>, /* Parent class */
                               Add,      /* Name of function in C++ (must match Python name) */
                               other_val);
    }
};

template <int SerNo>
int AddInCppRawPtr(const Abase<SerNo> *obj, int other_val) {
    return obj->Add(other_val) * 10 + 7;
}

template <int SerNo>
int AddInCppSharedPtr(std::shared_ptr<Abase<SerNo>> obj, int other_val) {
    return obj->Add(other_val) * 100 + 11;
}

template <int SerNo>
int AddInCppUniquePtr(std::unique_ptr<Abase<SerNo>> obj, int other_val) {
    return obj->Add(other_val) * 100 + 13;
}

template <int SerNo>
void wrap(py::module_ m, const char *py_class_name) {
    py::classh<Abase<SerNo>, AbaseAlias<SerNo>>(m, py_class_name)
        .def(py::init<int>(), py::arg("val"))
        .def("Get", &Abase<SerNo>::Get)
        .def("Add", &Abase<SerNo>::Add, py::arg("other_val"));

    m.def("AddInCppRawPtr", AddInCppRawPtr<SerNo>, py::arg("obj"), py::arg("other_val"));
    m.def("AddInCppSharedPtr", AddInCppSharedPtr<SerNo>, py::arg("obj"), py::arg("other_val"));
    m.def("AddInCppUniquePtr", AddInCppUniquePtr<SerNo>, py::arg("obj"), py::arg("other_val"));
}

} // namespace class_sh_trampoline_basic
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_trampoline_basic::Abase<0>)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_trampoline_basic::Abase<1>)

TEST_SUBMODULE(class_sh_trampoline_basic, m) {
    using namespace pybind11_tests::class_sh_trampoline_basic;
    wrap<0>(m, "Abase0");
    wrap<1>(m, "Abase1");
}
