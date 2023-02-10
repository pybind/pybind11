#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "pybind11_tests.h"

#include <array>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

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

using SetPairString = std::set<PairString>;

SetPairString return_set_pair_string() { return SetPairString({return_pair_string()}); }

using VectorPairString = std::vector<PairString>;

VectorPairString return_vector_pair_string() { return VectorPairString({return_pair_string()}); }

using ArrayPairString = std::array<PairString, 1>;

ArrayPairString return_array_pair_string() { return ArrayPairString({return_pair_string()}); }

using OptionalPairString = std::optional<PairString>;

OptionalPairString return_optional_pair_string() {
    return OptionalPairString(return_pair_string());
}

using VariantPairString = std::variant<PairString>;

VariantPairString return_variant_pair_string() { return VariantPairString(return_pair_string()); }

std::string call_callback_pass_pair_string(const std::function<std::string(PairString)> &cb) {
    auto p = return_pair_string();
    return cb(p);
}

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
        "return_map_sbbs",
        []() { return return_map_pair_string(); },
        py::return_value_policy_pack({{rvpc, rvpb}, {rvpb, rvpc}}));
    m.def(
        "return_map_bssb",
        []() { return return_map_pair_string(); },
        py::return_value_policy_pack({{rvpb, rvpc}, {rvpc, rvpb}}));

    m.def(
        "return_set_sb",
        []() { return return_set_pair_string(); },
        py::return_value_policy_pack({rvpc, rvpb}));
    m.def(
        "return_set_bs",
        []() { return return_set_pair_string(); },
        py::return_value_policy_pack({rvpb, rvpc}));

    m.def(
        "return_vector_sb",
        []() { return return_vector_pair_string(); },
        py::return_value_policy_pack({rvpc, rvpb}));
    m.def(
        "return_vector_bs",
        []() { return return_vector_pair_string(); },
        py::return_value_policy_pack({rvpb, rvpc}));

    m.def(
        "return_array_sb",
        []() { return return_array_pair_string(); },
        py::return_value_policy_pack({rvpc, rvpb}));
    m.def(
        "return_array_bs",
        []() { return return_array_pair_string(); },
        py::return_value_policy_pack({rvpb, rvpc}));

    m.def(
        "return_optional_sb",
        []() { return return_optional_pair_string(); },
        py::return_value_policy_pack({rvpc, rvpb}));
    m.def(
        "return_optional_bs",
        []() { return return_optional_pair_string(); },
        py::return_value_policy_pack({rvpb, rvpc}));

    m.def(
        "return_variant_sb",
        []() { return return_variant_pair_string(); },
        py::return_value_policy_pack({rvpc, rvpb}));
    m.def(
        "return_variant_bs",
        []() { return return_variant_pair_string(); },
        py::return_value_policy_pack({rvpb, rvpc}));

    // Here the rvp is applied to the return value of call_callback_pass_pair_string:
    m.def("call_callback_pass_pair_string_rtn_s", call_callback_pass_pair_string);
    m.def("call_callback_pass_pair_string_rtn_b", call_callback_pass_pair_string, rvpb);
}
