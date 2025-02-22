#include "pybind11_tests.h"

#include <memory>
#include <string>

namespace pybind11_tests {
namespace class_sh_factory_constructors {

template <int> // Using int as a trick to easily generate a series of types.
struct atyp {  // Short for "any type".
    std::string mtxt;
};

template <typename T>
std::string get_mtxt(const T &obj) {
    return obj.mtxt;
}

using atyp_valu = atyp<0x0>;
using atyp_rref = atyp<0x1>;
using atyp_cref = atyp<0x2>;
using atyp_mref = atyp<0x3>;
using atyp_cptr = atyp<0x4>;
using atyp_mptr = atyp<0x5>;
using atyp_shmp = atyp<0x6>;
using atyp_shcp = atyp<0x7>;
using atyp_uqmp = atyp<0x8>;
using atyp_uqcp = atyp<0x9>;
using atyp_udmp = atyp<0xA>;
using atyp_udcp = atyp<0xB>;

// clang-format off

atyp_valu        rtrn_valu() { atyp_valu obj{"Valu"}; return obj; }
atyp_rref&&      rtrn_rref() { static atyp_rref obj; obj.mtxt = "Rref"; return std::move(obj); }
atyp_cref const& rtrn_cref() { static atyp_cref obj; obj.mtxt = "Cref"; return obj; }
atyp_mref&       rtrn_mref() { static atyp_mref obj; obj.mtxt = "Mref"; return obj; }
atyp_cptr const* rtrn_cptr() { return new atyp_cptr{"Cptr"}; }
atyp_mptr*       rtrn_mptr() { return new atyp_mptr{"Mptr"}; }

std::shared_ptr<atyp_shmp>       rtrn_shmp() { return std::make_shared<atyp_shmp>(atyp_shmp{"Shmp"}); }
std::shared_ptr<atyp_shcp const> rtrn_shcp() { return std::shared_ptr<atyp_shcp const>(new atyp_shcp{"Shcp"}); }

std::unique_ptr<atyp_uqmp>       rtrn_uqmp() { return std::unique_ptr<atyp_uqmp      >(new atyp_uqmp{"Uqmp"}); }
std::unique_ptr<atyp_uqcp const> rtrn_uqcp() { return std::unique_ptr<atyp_uqcp const>(new atyp_uqcp{"Uqcp"}); }

struct sddm : std::default_delete<atyp_udmp      > {};
struct sddc : std::default_delete<atyp_udcp const> {};

std::unique_ptr<atyp_udmp,       sddm> rtrn_udmp() { return std::unique_ptr<atyp_udmp,       sddm>(new atyp_udmp{"Udmp"}); }
std::unique_ptr<atyp_udcp const, sddc> rtrn_udcp() { return std::unique_ptr<atyp_udcp const, sddc>(new atyp_udcp{"Udcp"}); }

// clang-format on

// Minimalistic approach to achieve full coverage of construct() overloads for constructing
// smart_holder from unique_ptr and shared_ptr returns.
struct with_alias {
    int val = 0;
    virtual ~with_alias() = default;
    // Some compilers complain about implicitly defined versions of some of the following:
    with_alias() = default;
    with_alias(const with_alias &) = default;
    with_alias(with_alias &&) = default;
    with_alias &operator=(const with_alias &) = default;
    with_alias &operator=(with_alias &&) = default;
};
struct with_alias_alias : with_alias {};
struct sddwaa : std::default_delete<with_alias_alias> {};

} // namespace class_sh_factory_constructors
} // namespace pybind11_tests

TEST_SUBMODULE(class_sh_factory_constructors, m) {
    using namespace pybind11_tests::class_sh_factory_constructors;

    py::classh<atyp_valu>(m, "atyp_valu")
        .def(py::init(&rtrn_valu))
        .def("get_mtxt", get_mtxt<atyp_valu>);

    py::classh<atyp_rref>(m, "atyp_rref")
        .def(py::init(&rtrn_rref))
        .def("get_mtxt", get_mtxt<atyp_rref>);

    py::classh<atyp_cref>(m, "atyp_cref")
        // class_: ... must return a compatible ...
        // classh: ... cannot pass object of non-trivial type ...
        // .def(py::init(&rtrn_cref))
        .def("get_mtxt", get_mtxt<atyp_cref>);

    py::classh<atyp_mref>(m, "atyp_mref")
        // class_: ... must return a compatible ...
        // classh: ... cannot pass object of non-trivial type ...
        // .def(py::init(&rtrn_mref))
        .def("get_mtxt", get_mtxt<atyp_mref>);

    py::classh<atyp_cptr>(m, "atyp_cptr")
        // class_: ... must return a compatible ...
        // classh: ... must return a compatible ...
        // .def(py::init(&rtrn_cptr))
        .def("get_mtxt", get_mtxt<atyp_cptr>);

    py::classh<atyp_mptr>(m, "atyp_mptr")
        .def(py::init(&rtrn_mptr))
        .def("get_mtxt", get_mtxt<atyp_mptr>);

    py::classh<atyp_shmp>(m, "atyp_shmp")
        .def(py::init(&rtrn_shmp))
        .def("get_mtxt", get_mtxt<atyp_shmp>);

    py::classh<atyp_shcp>(m, "atyp_shcp")
        // py::class_<atyp_shcp, std::shared_ptr<atyp_shcp>>(m, "atyp_shcp")
        // class_: ... must return a compatible ...
        // classh: ... cannot pass object of non-trivial type ...
        // .def(py::init(&rtrn_shcp))
        .def("get_mtxt", get_mtxt<atyp_shcp>);

    py::classh<atyp_uqmp>(m, "atyp_uqmp")
        .def(py::init(&rtrn_uqmp))
        .def("get_mtxt", get_mtxt<atyp_uqmp>);

    py::classh<atyp_uqcp>(m, "atyp_uqcp")
        // class_: ... cannot pass object of non-trivial type ...
        // classh: ... cannot pass object of non-trivial type ...
        // .def(py::init(&rtrn_uqcp))
        .def("get_mtxt", get_mtxt<atyp_uqcp>);

    py::classh<atyp_udmp>(m, "atyp_udmp")
        .def(py::init(&rtrn_udmp))
        .def("get_mtxt", get_mtxt<atyp_udmp>);

    py::classh<atyp_udcp>(m, "atyp_udcp")
        // py::class_<atyp_udcp, std::unique_ptr<atyp_udcp, sddc>>(m, "atyp_udcp")
        // class_: ... must return a compatible ...
        // classh: ... cannot pass object of non-trivial type ...
        // .def(py::init(&rtrn_udcp))
        .def("get_mtxt", get_mtxt<atyp_udcp>);

    py::classh<with_alias, with_alias_alias>(m, "with_alias")
        .def_readonly("val", &with_alias::val)
        .def(py::init([](int i) {
            auto p = std::unique_ptr<with_alias_alias, sddwaa>(new with_alias_alias);
            p->val = i * 100;
            return p;
        }))
        .def(py::init([](int i, int j) {
            auto p = std::unique_ptr<with_alias_alias>(new with_alias_alias);
            p->val = i * 100 + j * 10;
            return p;
        }))
        .def(py::init([](int i, int j, int k) {
            auto p = std::make_shared<with_alias_alias>();
            p->val = i * 100 + j * 10 + k;
            return p;
        }))
        .def(py::init(
            [](int, int, int, int) { return std::unique_ptr<with_alias>(new with_alias); },
            [](int, int, int, int) {
                return std::unique_ptr<with_alias>(new with_alias); // Invalid alias factory.
            }))
        .def(py::init([](int, int, int, int, int) { return std::make_shared<with_alias>(); },
                      [](int, int, int, int, int) {
                          return std::make_shared<with_alias>(); // Invalid alias factory.
                      }));
}
