#include <pybind11/stl.h>

#include "pybind11_tests.h"

#include <string>
#include <utility>

namespace {

using PairString = std::pair<std::string, std::string>;

PairString return_pair_string() { return PairString({"", ""}); }

using NestedPairString = std::pair<PairString, PairString>;

NestedPairString return_nested_pair_string() {
    return NestedPairString(return_pair_string(), return_pair_string());
}

using MapString = std::map<std::string, std::string>;

MapString return_map_string() { return MapString({return_pair_string()}); }

using MapPairString = std::map<PairString, PairString>;

MapPairString return_map_pair_string() { return MapPairString({return_nested_pair_string()}); }

} // namespace

TEST_SUBMODULE(return_value_policy_pack, m) {
    auto rvpc = py::return_value_policy::_clif_automatic;
    auto rvpb = py::return_value_policy::_return_as_bytes;

    m.def("return_tuple_str_str", []() { return return_pair_string(); });
    m.def(
        "return_tuple_bytes_bytes", []() { return return_pair_string(); }, rvpb);
    m.def(
        "return_tuple_str_bytes",
        []() { return return_pair_string(); },
        py::return_value_policy_pack({rvpc, rvpb}));
    m.def(
        "return_tuple_bytes_str",
        []() { return return_pair_string(); },
        py::return_value_policy_pack({rvpb, rvpc}));

    m.def("return_nested_tuple_str", []() { return return_nested_pair_string(); });
    m.def(
        "return_nested_tuple_bytes", []() { return return_nested_pair_string(); }, rvpb);
    m.def(
        "return_nested_tuple_str_bytes_bytes_str",
        []() { return return_nested_pair_string(); },
        py::return_value_policy_pack({{rvpc, rvpb}, {rvpb, rvpc}}));
    m.def(
        "return_nested_tuple_bytes_str_str_bytes",
        []() { return return_nested_pair_string(); },
        py::return_value_policy_pack({{rvpb, rvpc}, {rvpc, rvpb}}));

    m.def("return_dict_str_str", []() { return return_map_string(); });
    m.def(
        "return_dict_bytes_bytes", []() { return return_map_string(); }, rvpb);
    m.def(
        "return_dict_str_bytes",
        []() { return return_map_string(); },
        py::return_value_policy_pack({rvpc, rvpb}));
    m.def(
        "return_dict_bytes_str",
        []() { return return_map_string(); },
        py::return_value_policy_pack({rvpb, rvpc}));

    m.def(
        "return_dict_sbbs",
        []() { return return_map_pair_string(); },
        py::return_value_policy_pack({{rvpc, rvpb}, {rvpb, rvpc}}));
    m.def(
        "return_dict_bssb",
        []() { return return_map_pair_string(); },
        py::return_value_policy_pack({{rvpb, rvpc}, {rvpc, rvpb}}));
}
