#include <pybind11/smart_holder.h>

#include "pybind11_tests.h"

#include <memory>

namespace pybind11_tests {
namespace class_sh_inheritance {

template <int Id>
struct base_template {
    base_template() : base_id(Id) {}
    virtual ~base_template() = default;
    virtual int id() const { return base_id; }
    int base_id;

    // Some compilers complain about implicitly defined versions of some of the following:
    base_template(const base_template &) = default;
    base_template(base_template &&) noexcept = default;
    base_template &operator=(const base_template &) = default;
    base_template &operator=(base_template &&) noexcept = default;
};

using base = base_template<100>;

struct drvd : base {
    int id() const override { return 2 * base_id; }
};

// clang-format off
inline drvd *rtrn_mptr_drvd()         { return new drvd; }
inline base *rtrn_mptr_drvd_up_cast() { return new drvd; }

inline int pass_cptr_base(base const *b) { return b->id() + 11; }
inline int pass_cptr_drvd(drvd const *d) { return d->id() + 12; }

inline std::shared_ptr<drvd> rtrn_shmp_drvd()         { return std::make_shared<drvd>(); }
inline std::shared_ptr<base> rtrn_shmp_drvd_up_cast() { return std::make_shared<drvd>(); }

inline int pass_shcp_base(const std::shared_ptr<base const>& b) { return b->id() + 21; }
inline int pass_shcp_drvd(const std::shared_ptr<drvd const>& d) { return d->id() + 22; }
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

inline int pass_cptr_base1(base1 const *b) { return b->id() + 21; }
inline int pass_cptr_base2(base2 const *b) { return b->id() + 22; }
inline int pass_cptr_drvd2(drvd2 const *d) { return d->id() + 23; }
// clang-format on

} // namespace class_sh_inheritance
} // namespace pybind11_tests

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_inheritance::base)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_inheritance::drvd)

PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_inheritance::base1)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_inheritance::base2)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(pybind11_tests::class_sh_inheritance::drvd2)

namespace pybind11_tests {
namespace class_sh_inheritance {

TEST_SUBMODULE(class_sh_inheritance, m) {
    py::classh<base>(m, "base");
    py::classh<drvd, base>(m, "drvd");

    auto rvto = py::return_value_policy::take_ownership;

    m.def("rtrn_mptr_drvd", rtrn_mptr_drvd, rvto);
    m.def("rtrn_mptr_drvd_up_cast", rtrn_mptr_drvd_up_cast, rvto);
    m.def("pass_cptr_base", pass_cptr_base);
    m.def("pass_cptr_drvd", pass_cptr_drvd);

    m.def("rtrn_shmp_drvd", rtrn_shmp_drvd);
    m.def("rtrn_shmp_drvd_up_cast", rtrn_shmp_drvd_up_cast);
    m.def("pass_shcp_base", pass_shcp_base);
    m.def("pass_shcp_drvd", pass_shcp_drvd);

    // __init__ needed for Python inheritance.
    py::classh<base1>(m, "base1").def(py::init<>());
    py::classh<base2>(m, "base2").def(py::init<>());
    py::classh<drvd2, base1, base2>(m, "drvd2");

    m.def("rtrn_mptr_drvd2", rtrn_mptr_drvd2, rvto);
    m.def("rtrn_mptr_drvd2_up_cast1", rtrn_mptr_drvd2_up_cast1, rvto);
    m.def("rtrn_mptr_drvd2_up_cast2", rtrn_mptr_drvd2_up_cast2, rvto);
    m.def("pass_cptr_base1", pass_cptr_base1);
    m.def("pass_cptr_base2", pass_cptr_base2);
    m.def("pass_cptr_drvd2", pass_cptr_drvd2);
}

} // namespace class_sh_inheritance
} // namespace pybind11_tests
