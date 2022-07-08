#include "pybind11/pybind11.h"

PYBIND11_MODULE(namespace_visibility_2, m) { m.doc() = "ns_vis_2"; }
