#include "pybind11_tests.h"

#include "pybind11/smart_holder.h"

#include <memory>
#include <vector>

namespace {

struct DD {
    int i;
};

struct CC {
    int i = 13;
};

struct BB {
    CC c;
    DD d;

    DD GetD() { return d; }

    void SetD(const DD &dd) { d = dd; }
};

struct AA {
    BB b;
    CC c;
    DD *dp;
    std::unique_ptr<DD> du;
    std::shared_ptr<DD> ds;
    std::vector<int> v;
};

inline void ConsumeCC(std::unique_ptr<CC>) {}
inline void ConsumeAA(std::unique_ptr<AA>) {}

} // namespace

PYBIND11_SMART_HOLDER_TYPE_CASTERS(DD)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(CC)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(BB)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(AA)

TEST_SUBMODULE(class_sh_nested_fields, m) {
    m.def("to_cout", to_cout);

    py::classh<::DD> DD_class(m, "DD");
    DD_class.def_readwrite("i", &::DD::i);
    DD_class.def(py::init<>());

    py::classh<::CC> CC_class(m, "CC");
    CC_class.def_readwrite("i", &::CC::i);
    CC_class.def(py::init<>());

    py::classh<::BB> BB_class(m, "BB");
    BB_class.def_readwrite("c", &::BB::c);
    BB_class.def_property("d", &::BB::GetD, &::BB::SetD);
    BB_class.def(py::init<>());

    py::classh<::AA> AA_class(m, "AA");
    AA_class.def_readwrite("b", &::AA::b);
    AA_class.def_readwrite("v", &::AA::v);
    AA_class.def("GetC", [](::AA &self) { return self.c; });
    AA_class.def("SetC", [](::AA &self, ::CC v) { self.c = v; });
    AA_class.def_property(
        "dp",
        [](::AA &self) { return *self.dp; },
        [](::AA &self, ::DD *v) { self.dp = v; },
        py::return_value_policy::reference_internal);
    AA_class.def_property(
        "du",
        [](::AA &self) { return *self.du; },
        [](::AA &self, ::std::unique_ptr<::DD> v) { self.du = std::move(v); },
        py::return_value_policy::reference_internal);
    AA_class.def_readwrite("ds", &::AA::ds);
    AA_class.def(py::init<>());

    m.def("ConsumeCC",
          (void (*)(::std::unique_ptr<::CC>)) & ::ConsumeCC,
          py::arg("cc"),
          py::return_value_policy::automatic);

    m.def("ConsumeAA",
          (void (*)(::std::unique_ptr<::AA>)) & ::ConsumeAA,
          py::arg("aa"),
          py::return_value_policy::automatic);
}
