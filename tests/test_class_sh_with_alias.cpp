#include "pybind11_tests.h"

#include <pybind11/smart_holder.h>

#include <cstdint>
#include <memory>

namespace pybind11_tests {
namespace test_class_sh_with_alias {

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

struct Passenger {
    std::string mtxt;
    // on construction: store pointer as an id
    Passenger() : mtxt(id() + "_") {}
    Passenger(const Passenger &other) { mtxt = other.mtxt + "Copy->" + id(); }
    Passenger(Passenger &&other) { mtxt = other.mtxt + "Move->" + id(); }
    std::string id() const { return std::to_string(reinterpret_cast<uintptr_t>(this)); }
};
struct ConsumerBase {
    ConsumerBase()                     = default;
    ConsumerBase(const ConsumerBase &) = default;
    ConsumerBase(ConsumerBase &&)      = default;
    virtual ~ConsumerBase()            = default;
    virtual void pass_uq_cref(const std::unique_ptr<Passenger> &obj) { modify(*obj); };
    virtual void pass_valu(Passenger obj) { modify(obj); };
    virtual void pass_lref(Passenger &obj) { modify(obj); };
    virtual void pass_cref(const Passenger &obj) { modify(const_cast<Passenger &>(obj)); };
    void modify(Passenger &obj) {
        // when base virtual method is called: append obj pointer again (should be same as before)
        obj.mtxt.append("_");
        obj.mtxt.append(std::to_string(reinterpret_cast<uintptr_t>(&obj)));
    }
};
struct ConsumerBaseAlias : ConsumerBase {
    using ConsumerBase::ConsumerBase;
    void pass_uq_cref(const std::unique_ptr<Passenger> &obj) override {
        PYBIND11_OVERRIDE(void, ConsumerBase, pass_uq_cref, obj);
    }
    void pass_valu(Passenger obj) override {
        PYBIND11_OVERRIDE(void, ConsumerBase, pass_valu, obj);
    }
    void pass_lref(Passenger &obj) override {
        PYBIND11_OVERRIDE(void, ConsumerBase, pass_lref, obj);
    }
    void pass_cref(const Passenger &obj) override {
        PYBIND11_OVERRIDE(void, ConsumerBase, pass_cref, obj);
    }
};

// check roundtrip of Passenger send to ConsumerBaseAlias
// TODO: Find template magic to avoid code duplication
std::string check_roundtrip_uq_cref(ConsumerBase &consumer) {
    std::unique_ptr<Passenger> obj(new Passenger());
    consumer.pass_uq_cref(obj);
    return obj->mtxt;
}
std::string check_roundtrip_valu(ConsumerBase &consumer) {
    Passenger obj;
    consumer.pass_valu(obj);
    return obj.mtxt;
}
std::string check_roundtrip_lref(ConsumerBase &consumer) {
    Passenger obj;
    consumer.pass_lref(obj);
    return obj.mtxt;
}
std::string check_roundtrip_cref(ConsumerBase &consumer) {
    Passenger obj;
    consumer.pass_cref(obj);
    return obj.mtxt;
}

} // namespace test_class_sh_with_alias
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::test_class_sh_with_alias::Abase<0>)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::test_class_sh_with_alias::Abase<1>)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::test_class_sh_with_alias::Passenger)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::test_class_sh_with_alias::ConsumerBase)

TEST_SUBMODULE(class_sh_with_alias, m) {
    using namespace pybind11_tests::test_class_sh_with_alias;
    wrap<0>(m, "Abase0");
    wrap<1>(m, "Abase1");

    py::classh<Passenger>(m, "Passenger").def_readwrite("mtxt", &Passenger::mtxt);

    py::classh<ConsumerBase, ConsumerBaseAlias>(m, "ConsumerBase")
        .def(py::init<>())
        .def("pass_uq_cref", &ConsumerBase::pass_uq_cref)
        .def("pass_valu", &ConsumerBase::pass_valu)
        .def("pass_lref", &ConsumerBase::pass_lref)
        .def("pass_cref", &ConsumerBase::pass_cref);

    m.def("check_roundtrip_uq_cref", check_roundtrip_uq_cref);
    m.def("check_roundtrip_valu", check_roundtrip_valu);
    m.def("check_roundtrip_lref", check_roundtrip_lref);
    m.def("check_roundtrip_cref", check_roundtrip_cref);
}
