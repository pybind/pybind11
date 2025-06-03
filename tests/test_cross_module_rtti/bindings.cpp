#include <pybind11/pybind11.h>

#include <lib.h>

class BaseTrampoline : public lib::Base, public pybind11::trampoline_self_life_support {
public:
    using lib::Base::Base;
    int get() const override { PYBIND11_OVERLOAD(int, lib::Base, get); }
};

PYBIND11_MODULE(test_cross_module_rtti_bindings, m) {
    pybind11::classh<lib::Base, BaseTrampoline>(m, "Base")
        .def(pybind11::init<int, int>())
        .def_readwrite("a", &lib::Base::a)
        .def_readwrite("b", &lib::Base::b);

    m.def("get_foo", [](int a, int b) -> std::shared_ptr<lib::Base> {
        return std::make_shared<lib::Foo>(a, b);
    });
}
