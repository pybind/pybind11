#include "pybind11_tests.h"

#include <pybind11/classh.h>

namespace pybind11_tests {
namespace classh_inheritance {

template <int Id>
struct base_template {
    base_template() : base_id(Id) {}
    virtual ~base_template() = default;
    virtual int id() const { return base_id; }
    int base_id;
};

using base = base_template<100>;

struct drvd : base {
    int id() const override { return 2 * base_id; }
};

// clang-format off
inline drvd *rtrn_mptr_drvd()         { return new drvd; }
inline base *rtrn_mptr_drvd_up_cast() { return new drvd; }

inline int pass_cptr_base(const base *b) { return b->id() + 11; }
inline int pass_cptr_drvd(const drvd *d) { return d->id() + 12; }
// clang-format on

using base1 = base_template<110>;
using base2 = base_template<120>;

// Not reusing base here because it would interfere with the single-inheritance test.
struct drvd2 : base1, base2 {
    int id() const override { return 3 * base1::base_id + 4 * base2::base_id; }
};

// clang-format off
inline drvd2 *rtrn_mptr_drvd2()          { return new drvd2; }
inline base1 *rtrn_mptr_drvd2_up_cast1() { return new drvd2; }
inline base2 *rtrn_mptr_drvd2_up_cast2() { return new drvd2; }

inline int pass_cptr_base1(const base1 *b) { return b->id() + 21; }
inline int pass_cptr_base2(const base2 *b) { return b->id() + 22; }
inline int pass_cptr_drvd2(const drvd2 *d) { return d->id() + 23; }
// clang-format on

} // namespace classh_inheritance
} // namespace pybind11_tests

PYBIND11_CLASSH_TYPE_CASTERS(pybind11_tests::classh_inheritance::base)
PYBIND11_CLASSH_TYPE_CASTERS(pybind11_tests::classh_inheritance::drvd)

PYBIND11_CLASSH_TYPE_CASTERS(pybind11_tests::classh_inheritance::base1)
PYBIND11_CLASSH_TYPE_CASTERS(pybind11_tests::classh_inheritance::base2)
PYBIND11_CLASSH_TYPE_CASTERS(pybind11_tests::classh_inheritance::drvd2)

namespace pybind11_tests {
namespace classh_inheritance {

TEST_SUBMODULE(classh_inheritance, m) {
    py::classh<base>(m, "base");
    py::classh<drvd, base>(m, "drvd");

    m.def("rtrn_mptr_drvd", rtrn_mptr_drvd, py::return_value_policy::take_ownership);
    m.def(
        "rtrn_mptr_drvd_up_cast", rtrn_mptr_drvd_up_cast, py::return_value_policy::take_ownership);
    m.def("pass_cptr_base", pass_cptr_base);
    m.def("pass_cptr_drvd", pass_cptr_drvd);

    py::classh<base1>(m, "base1").def(py::init<>()); // __init__ needed for Python inheritance.
    py::classh<base2>(m, "base2").def(py::init<>());
    py::classh<drvd2, base1, base2>(m, "drvd2");

    m.def("rtrn_mptr_drvd2", rtrn_mptr_drvd2, py::return_value_policy::take_ownership);
    m.def("rtrn_mptr_drvd2_up_cast1",
          rtrn_mptr_drvd2_up_cast1,
          py::return_value_policy::take_ownership);
    m.def("rtrn_mptr_drvd2_up_cast2",
          rtrn_mptr_drvd2_up_cast2,
          py::return_value_policy::take_ownership);
    m.def("pass_cptr_base1", pass_cptr_base1);
    m.def("pass_cptr_base2", pass_cptr_base2);
    m.def("pass_cptr_drvd2", pass_cptr_drvd2);
}

} // namespace classh_inheritance
} // namespace pybind11_tests
