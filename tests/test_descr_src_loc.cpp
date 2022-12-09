// Copyright (c) 2022 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#include "pybind11_tests.h"

// This test actually works with almost all C++17 compilers, but is currently
// only needed (and tested) for type_caster_odr_guard.h, for simplicity.

#ifndef PYBIND11_ENABLE_TYPE_CASTER_ODR_GUARD

TEST_SUBMODULE(descr_src_loc, m) { m.attr("block_descr_offset") = py::none(); }

#else

namespace pybind11_tests {
namespace descr_src_loc {

using py::detail::const_name;
using py::detail::src_loc;

struct block_descr {
    static constexpr unsigned offset = __LINE__;
    static constexpr auto c0 = py::detail::descr<0>(src_loc::here());
    static constexpr auto c1 = py::detail::descr<3>("Abc");
    static constexpr auto c2 = py::detail::descr<1>(src_loc::here(), 'D');
    static constexpr auto c3 = py::detail::descr<2>(src_loc::here(), 'E', 'f');
};

struct block_const_name {
    static constexpr unsigned offset = __LINE__;
    static constexpr auto c0 = const_name("G");
    static constexpr auto c1 = const_name("Hi");
    static constexpr auto c2 = const_name<0>();
    static constexpr auto c3 = const_name<1>();
    static constexpr auto c4 = const_name<23>();
    static constexpr auto c5 = const_name<std::string>();
    static constexpr auto c6 = const_name<true>("J", "K");
    static constexpr auto c7 = const_name<false>("L", "M");
};

#    if defined(PYBIND11_DETAIL_UNDERSCORE_BACKWARD_COMPATIBILITY)
struct block_underscore {
    static constexpr unsigned offset = __LINE__;
    // Using a macro to avoid copying the block_const_name code garbles the src_loc.line numbers.
    static constexpr auto c0 = const_name("G");
    static constexpr auto c1 = const_name("Hi");
    static constexpr auto c2 = const_name<0>();
    static constexpr auto c3 = const_name<1>();
    static constexpr auto c4 = const_name<23>();
    static constexpr auto c5 = const_name<std::string>();
    static constexpr auto c6 = const_name<true>("J", "K");
    static constexpr auto c7 = const_name<false>("L", "M");
};
#    endif

struct block_plus {
    static constexpr unsigned offset = __LINE__;
    static constexpr auto c0 = const_name("N") + // critical line break
                               const_name("O");
    static constexpr auto c1 = const_name("P", src_loc(nullptr, 0)) + // critical line break
                               const_name("Q");
};

struct block_concat {
    static constexpr unsigned offset = __LINE__;
    static constexpr auto c0 = py::detail::concat(const_name("R"));
    static constexpr auto c1 = py::detail::concat(const_name("S"), // critical line break
                                                  const_name("T"));
    static constexpr auto c2
        = py::detail::concat(const_name("U", src_loc(nullptr, 0)), // critical line break
                             const_name("V"));
};

struct block_type_descr {
    static constexpr unsigned offset = __LINE__;
    static constexpr auto c0 = py::detail::type_descr(const_name("W"));
};

struct block_int_to_str {
    static constexpr unsigned offset = __LINE__;
    static constexpr auto c0 = py::detail::int_to_str<0>::digits;
    static constexpr auto c1 = py::detail::int_to_str<4>::digits;
    static constexpr auto c2 = py::detail::int_to_str<56>::digits;
};

} // namespace descr_src_loc
} // namespace pybind11_tests

TEST_SUBMODULE(descr_src_loc, m) {
    using namespace pybind11_tests::descr_src_loc;

#    define ATTR_OFFS(B) m.attr(#B "_offset") = B::offset;
#    define ATTR_BLKC(B, C)                                                                       \
        m.attr(#B "_" #C) = py::make_tuple(B::C.text, B::C.sloc.file, B::C.sloc.line);

    ATTR_OFFS(block_descr)
    ATTR_BLKC(block_descr, c0)
    ATTR_BLKC(block_descr, c1)
    ATTR_BLKC(block_descr, c2)
    ATTR_BLKC(block_descr, c3)

    ATTR_OFFS(block_const_name)
    ATTR_BLKC(block_const_name, c0)
    ATTR_BLKC(block_const_name, c1)
    ATTR_BLKC(block_const_name, c2)
    ATTR_BLKC(block_const_name, c3)
    ATTR_BLKC(block_const_name, c4)
    ATTR_BLKC(block_const_name, c5)
    ATTR_BLKC(block_const_name, c6)
    ATTR_BLKC(block_const_name, c7)

    ATTR_OFFS(block_underscore)
    ATTR_BLKC(block_underscore, c0)
    ATTR_BLKC(block_underscore, c1)
    ATTR_BLKC(block_underscore, c2)
    ATTR_BLKC(block_underscore, c3)
    ATTR_BLKC(block_underscore, c4)
    ATTR_BLKC(block_underscore, c5)
    ATTR_BLKC(block_underscore, c6)
    ATTR_BLKC(block_underscore, c7)

    ATTR_OFFS(block_plus)
    ATTR_BLKC(block_plus, c0)
    ATTR_BLKC(block_plus, c1)

    ATTR_OFFS(block_concat)
    ATTR_BLKC(block_concat, c0)
    ATTR_BLKC(block_concat, c1)
    ATTR_BLKC(block_concat, c2)

    ATTR_OFFS(block_type_descr)
    ATTR_BLKC(block_type_descr, c0)

    ATTR_OFFS(block_int_to_str)
    ATTR_BLKC(block_int_to_str, c0)
    ATTR_BLKC(block_int_to_str, c1)
    ATTR_BLKC(block_int_to_str, c2)
}

#endif // PYBIND11_ENABLE_TYPE_CASTER_ODR_GUARD
