#include "pybind11_tests.h"

#include <memory>

namespace pybind11_tests {
namespace class_sh_disowning {

template <int SerNo> // Using int as a trick to easily generate a series of types.
struct Atype {
    int val = 0;
    explicit Atype(int val_) : val{val_} {}
    int get() const { return val * 10 + SerNo; }
};

int same_twice(std::unique_ptr<Atype<1>> at1a, std::unique_ptr<Atype<1>> at1b) {
    return at1a->get() * 100 + at1b->get() * 10;
}

int mixed(std::unique_ptr<Atype<1>> at1, std::unique_ptr<Atype<2>> at2) {
    return at1->get() * 200 + at2->get() * 20;
}

int overloaded(std::unique_ptr<Atype<1>> at1, int i) { return at1->get() * 30 + i; }
int overloaded(std::unique_ptr<Atype<2>> at2, int i) { return at2->get() * 40 + i; }

} // namespace class_sh_disowning
} // namespace pybind11_tests

TEST_SUBMODULE(class_sh_disowning, m) {
    using namespace pybind11_tests::class_sh_disowning;

    py::classh<Atype<1>>(m, "Atype1").def(py::init<int>()).def("get", &Atype<1>::get);
    py::classh<Atype<2>>(m, "Atype2").def(py::init<int>()).def("get", &Atype<2>::get);

    m.def("same_twice", same_twice);

    m.def("mixed", mixed);

    m.def("overloaded", (int (*)(std::unique_ptr<Atype<1>>, int)) &overloaded);
    m.def("overloaded", (int (*)(std::unique_ptr<Atype<2>>, int)) &overloaded);
}
