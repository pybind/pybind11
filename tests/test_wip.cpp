#include <pybind11/smart_holder.h>

#include "pybind11_tests.h"

namespace pybind11_tests {
namespace wip {

template <int SerNo> // Using int as a trick to easily generate a series of types.
struct Atype {
    int val = 0;
    explicit Atype(int val_) : val{val_} {}
    int get() const { return val * 10 + SerNo; }
};

int mixed(std::unique_ptr<Atype<1>> at1, std::unique_ptr<Atype<2>> at2) {
    return at1->get() * 200 + at2->get() * 20;
}

} // namespace wip
} // namespace pybind11_tests

using namespace pybind11_tests::wip;

PYBIND11_SMART_HOLDER_TYPE_CASTERS(Atype<1>)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(Atype<2>)

TEST_SUBMODULE(wip, m) {
    py::classh<Atype<1>>(m, "Atype1").def(py::init<int>()).def("get", &Atype<1>::get);
    py::classh<Atype<2>>(m, "Atype2").def(py::init<int>()).def("get", &Atype<2>::get);

    m.def("mixed", mixed);
}
