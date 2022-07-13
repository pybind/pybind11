// Copyright (c) 2022 The Pybind Development Team.
// All rights reserved. Use of this source code is governed by a
// BSD-style license that can be found in the LICENSE file.

#pragma once

#include <cstddef>

#ifdef __GNUG__
#    define PYBIND11_NS_VIS_U /* unspecified */
#    define PYBIND11_NS_VIS_H __attribute__((visibility("hidden")))
#else
#    define PYBIND11_NS_VIS_U
#    define PYBIND11_NS_VIS_H
#endif

#define PYBIND11_NS_VIS_FUNC                                                                      \
    inline std::ptrdiff_t func() {                                                                \
        static std::ptrdiff_t value = 0;                                                          \
        return reinterpret_cast<std::ptrdiff_t>(&value);                                          \
    }

#define PYBIND11_NS_VIS_DEFS                                                                      \
    m.def("ns_vis_uuu_func", pybind11_ns_vis_uuu::func);                                          \
    m.def("ns_vis_uuh_func", pybind11_ns_vis_uuh::func);                                          \
    m.def("ns_vis_uhu_func", pybind11_ns_vis_uhu::func);                                          \
    m.def("ns_vis_uhh_func", pybind11_ns_vis_uhh::func);                                          \
    m.def("ns_vis_huu_func", pybind11_ns_vis_huu::func);                                          \
    m.def("ns_vis_huh_func", pybind11_ns_vis_huh::func);                                          \
    m.def("ns_vis_hhu_func", pybind11_ns_vis_hhu::func);                                          \
    m.def("ns_vis_hhh_func", pybind11_ns_vis_hhh::func);
