#include "pybind11_tests.h"

#include <pybind11/classh.h>

namespace pybind11_tests {
namespace classh_inheritance {

struct base {
    base() : base_id(100) {}
    virtual ~base() = default;
    virtual int id() const { return base_id; }
    int base_id;
};

struct drvd : base {
    int id() const override { return 2 * base_id; }
};

inline drvd *make_drvd() { return new drvd; }
inline base *make_drvd_up_cast() { return new drvd; }

inline int pass_base(const base *b) { return b->id(); }
inline int pass_drvd(const drvd *d) { return d->id(); }

} // namespace classh_inheritance
} // namespace pybind11_tests

PYBIND11_CLASSH_TYPE_CASTERS(pybind11_tests::classh_inheritance::base)
PYBIND11_CLASSH_TYPE_CASTERS(pybind11_tests::classh_inheritance::drvd)

namespace pybind11_tests {
namespace classh_inheritance {

TEST_SUBMODULE(classh_inheritance, m) {
    py::classh<base>(m, "base");
    py::classh<drvd, base>(m, "drvd");

    m.def("make_drvd", make_drvd, py::return_value_policy::take_ownership);
    m.def("make_drvd_up_cast", make_drvd_up_cast, py::return_value_policy::take_ownership);
    m.def("pass_base", pass_base);
    m.def("pass_drvd", pass_drvd);
}

} // namespace classh_inheritance
} // namespace pybind11_tests
