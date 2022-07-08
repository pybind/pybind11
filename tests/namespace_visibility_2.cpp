#include "pybind11/pybind11.h"
#include "namespace_visibility.h"

#ifdef NEVER_DEFINED
// clang-format off
namespace pybind11_ns_vis_uuu PYBIND11_NS_VIS_U { PYBIND11_NS_VIS_FUNC }
namespace pybind11_ns_vis_uuh PYBIND11_NS_VIS_H { PYBIND11_NS_VIS_FUNC }
namespace pybind11_ns_vis_uhu PYBIND11_NS_VIS_U { PYBIND11_NS_VIS_FUNC }
namespace pybind11_ns_vis_uhh PYBIND11_NS_VIS_H { PYBIND11_NS_VIS_FUNC }
namespace pybind11_ns_vis_huu PYBIND11_NS_VIS_U { PYBIND11_NS_VIS_FUNC }
namespace pybind11_ns_vis_huh PYBIND11_NS_VIS_H { PYBIND11_NS_VIS_FUNC }
namespace pybind11_ns_vis_hhu PYBIND11_NS_VIS_U { PYBIND11_NS_VIS_FUNC }
namespace pybind11_ns_vis_hhh PYBIND11_NS_VIS_H { PYBIND11_NS_VIS_FUNC }
//                          ^                 ^
//                   bit used ............ here
// clang-format on
#endif

PYBIND11_MODULE(namespace_visibility_2, m) {
    m.doc() = "DISABLED";

    // PYBIND11_NS_VIS_DEFS
}
