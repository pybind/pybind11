#include "pybind11_tests.h"

#include <pybind11/smart_holder.h>

#include <cstdint>
#include <memory>

namespace pybind11_tests {
namespace class_sh_with_alias {

template <int SerNo> // Using int as a trick to easily generate a series of types.
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
struct AbaseAlias<1> : Abase<1>, py::virtual_overrider_self_life_support {
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

/* Tests passing objects by reference to Python-derived class methods */
struct Passenger { // object class passed around, recording its copy and move constructions
    std::string mtxt;
    Passenger() = default;
    // on copy or move: keep old mtxt and augment operation as well as new pointer id
    Passenger(const Passenger &other) { mtxt = other.mtxt + "Copy->" + std::to_string(id()); }
    Passenger(Passenger &&other) { mtxt = other.mtxt + "Move->" + std::to_string(id()); }
    uintptr_t id() const { return reinterpret_cast<uintptr_t>(this); }
};
struct ReferencePassingTest { // virtual base class used to test reference passing
    ReferencePassingTest()                             = default;
    ReferencePassingTest(const ReferencePassingTest &) = default;
    ReferencePassingTest(ReferencePassingTest &&)      = default;
    virtual ~ReferencePassingTest()                    = default;
    virtual uintptr_t pass_uq_cref(const std::unique_ptr<Passenger> &obj) { return modify(*obj); };
    // NOLINTNEXTLINE(clang-analyzer-core.StackAddrEscapeBase)
    virtual uintptr_t pass_valu(Passenger obj) { return modify(obj); };
    virtual uintptr_t pass_mref(Passenger &obj) { return modify(obj); };
    virtual uintptr_t pass_mptr(Passenger *obj) { return modify(*obj); };
    virtual uintptr_t pass_cref(const Passenger &obj) { return modify(obj); };
    virtual uintptr_t pass_cptr(const Passenger *obj) { return modify(*obj); };
    uintptr_t modify(const Passenger &obj) { return obj.id(); }
    uintptr_t modify(Passenger &obj) {
        obj.mtxt.append("_MODIFIED");
        return obj.id();
    }
};
struct PyReferencePassingTest : ReferencePassingTest {
    using ReferencePassingTest::ReferencePassingTest;
    uintptr_t pass_uq_cref(const std::unique_ptr<Passenger> &obj) override {
        PYBIND11_OVERRIDE(uintptr_t, ReferencePassingTest, pass_uq_cref, obj);
    }
    uintptr_t pass_valu(Passenger obj) override {
        PYBIND11_OVERRIDE(uintptr_t, ReferencePassingTest, pass_valu, obj);
    }
    uintptr_t pass_mref(Passenger &obj) override {
        PYBIND11_OVERRIDE(uintptr_t, ReferencePassingTest, pass_mref, obj);
    }
    uintptr_t pass_cref(const Passenger &obj) override {
        PYBIND11_OVERRIDE(uintptr_t, ReferencePassingTest, pass_cref, obj);
    }
    uintptr_t pass_mptr(Passenger *obj) override {
        PYBIND11_OVERRIDE(uintptr_t, ReferencePassingTest, pass_mptr, obj);
    }
    uintptr_t pass_cptr(const Passenger *obj) override {
        PYBIND11_OVERRIDE(uintptr_t, ReferencePassingTest, pass_cptr, obj);
    }
};

std::string evaluate(const Passenger &orig, uintptr_t cycled) {
    return orig.mtxt + (orig.id() == cycled ? "_REF" : "_COPY");
}
// Functions triggering virtual-method calls from python-derived class (caller)
// Goal: modifications to Passenger happening in Python-code methods
//       overriding the C++ virtual methods, should remain visible in C++.
std::string check_roundtrip_uq_cref(ReferencePassingTest &caller) {
    std::unique_ptr<Passenger> obj(new Passenger());
    return evaluate(*obj, caller.pass_uq_cref(obj));
}
// TODO: Find template magic to avoid this code duplication
std::string check_roundtrip_valu(ReferencePassingTest &caller) {
    Passenger obj;
    return evaluate(obj, caller.pass_valu(obj));
}
std::string check_roundtrip_mref(ReferencePassingTest &caller) {
    Passenger obj;
    return evaluate(obj, caller.pass_mref(obj));
}
std::string check_roundtrip_cref(ReferencePassingTest &caller) {
    Passenger obj;
    return evaluate(obj, caller.pass_cref(obj));
}
std::string check_roundtrip_mptr(ReferencePassingTest &caller) {
    Passenger obj;
    return evaluate(obj, caller.pass_mptr(&obj));
}
std::string check_roundtrip_cptr(ReferencePassingTest &caller) {
    Passenger obj;
    return evaluate(obj, caller.pass_cptr(&obj));
}

} // namespace class_sh_with_alias
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_with_alias::Abase<0>)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_with_alias::Abase<1>)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_with_alias::Passenger)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_with_alias::ReferencePassingTest)

TEST_SUBMODULE(class_sh_with_alias, m) {
    using namespace pybind11_tests::class_sh_with_alias;
    wrap<0>(m, "Abase0");
    wrap<1>(m, "Abase1");

    py::classh<Passenger>(m, "Passenger")
        .def_property_readonly("id", &Passenger::id)
        .def_readwrite("mtxt", &Passenger::mtxt);

    py::classh<ReferencePassingTest, PyReferencePassingTest>(m, "ReferencePassingTest")
        .def(py::init<>())
        .def("pass_uq_cref", &ReferencePassingTest::pass_uq_cref)
        .def("pass_valu", &ReferencePassingTest::pass_valu)
        .def("pass_mref", &ReferencePassingTest::pass_mref)
        .def("pass_cref", &ReferencePassingTest::pass_cref)
        .def("pass_mptr", &ReferencePassingTest::pass_mptr)
        .def("pass_cptr", &ReferencePassingTest::pass_cptr);

    m.def("check_roundtrip_uq_cref", check_roundtrip_uq_cref);
    m.def("check_roundtrip_valu", check_roundtrip_valu);
    m.def("check_roundtrip_mref", check_roundtrip_mref);
    m.def("check_roundtrip_cref", check_roundtrip_cref);
    m.def("check_roundtrip_mptr", check_roundtrip_mptr);
    m.def("check_roundtrip_cptr", check_roundtrip_cptr);
}
