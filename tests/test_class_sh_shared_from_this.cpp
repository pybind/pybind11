#include "pybind11_tests.h"
#include "object.h"

#include <pybind11/smart_holder.h>

namespace test_class_sh_shared_from_this {

class MyObject3 : public std::enable_shared_from_this<MyObject3> {
public:
    MyObject3(const MyObject3 &) = default;
    MyObject3(int value) : value(value) { print_created(this, toString()); }
    std::string toString() const { return "MyObject3[" + std::to_string(value) + "]"; }
    virtual ~MyObject3() { print_destroyed(this); }
private:
    int value;
};

struct SharedFromThisRef {
    struct B : std::enable_shared_from_this<B> {
        B() { print_created(this); }
        B(const B &) : std::enable_shared_from_this<B>() { print_copy_created(this); }
        B(B &&) : std::enable_shared_from_this<B>() { print_move_created(this); }
        ~B() { print_destroyed(this); }
    };

    B value = {};
    std::shared_ptr<B> shared = std::make_shared<B>();
};

struct SharedFromThisVBase : std::enable_shared_from_this<SharedFromThisVBase> {
    SharedFromThisVBase() = default;
    SharedFromThisVBase(const SharedFromThisVBase &) = default;
    virtual ~SharedFromThisVBase() = default;
};

struct SharedFromThisVirt : virtual SharedFromThisVBase {};

} // namespace test_class_sh_shared_from_this

using namespace test_class_sh_shared_from_this;

PYBIND11_SMART_HOLDER_TYPE_CASTERS(MyObject3)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(SharedFromThisRef::B)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(SharedFromThisRef)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(SharedFromThisVirt)

TEST_SUBMODULE(class_sh_shared_from_this, m) {
    // py::classh<MyObject3>(m, "MyObject3")
    //     .def(py::init<int>());
    m.def("make_myobject3_1", []() { return new MyObject3(8); });
    m.def("make_myobject3_2", []() { return std::make_shared<MyObject3>(9); });
    m.def("print_myobject3_1", [](const MyObject3 *obj) { py::print(obj->toString()); });
    m.def("print_myobject3_2", [](std::shared_ptr<MyObject3> obj) { py::print(obj->toString()); });
    m.def("print_myobject3_3", [](const std::shared_ptr<MyObject3> &obj) { py::print(obj->toString()); });
    // m.def("print_myobject3_4", [](const std::shared_ptr<MyObject3> *obj) { py::print((*obj)->toString()); });

    using B = SharedFromThisRef::B;
    // py::classh<B>(m, "B");
    py::classh<SharedFromThisRef>(m, "SharedFromThisRef")
        .def(py::init<>())
        .def_readonly("bad_wp", &SharedFromThisRef::value)
        .def_property_readonly("ref", [](const SharedFromThisRef &s) -> const B & { return *s.shared; })
        .def_property_readonly("copy", [](const SharedFromThisRef &s) { return s.value; },
                               py::return_value_policy::automatic) // XXX XXX XXX copy)
        .def_readonly("holder_ref", &SharedFromThisRef::shared)
        .def_property_readonly("holder_copy", [](const SharedFromThisRef &s) { return s.shared; },
                               py::return_value_policy::automatic) // XXX XXX XXX copy)
        .def("set_ref", [](SharedFromThisRef &, const B &) { return true; })
        .def("set_holder", [](SharedFromThisRef &, std::shared_ptr<B>) { return true; });

    static std::shared_ptr<SharedFromThisVirt> sft(new SharedFromThisVirt());
    // py::classh<SharedFromThisVirt>(m, "SharedFromThisVirt")
    //     .def_static("get", []() { return sft.get(); }, py::return_value_policy::reference);
}
