/*
    tests/test_stl.cpp -- STL type casters

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include <pybind11/stl.h>

/// Issue #528: templated constructor
struct TplCtorClass {
    template <typename T> TplCtorClass(const T &) { }
    bool operator==(const TplCtorClass &) const { return true; }
};

namespace std {
    template <>
    struct hash<TplCtorClass> { size_t operator()(const TplCtorClass &) const { return 0; } };
}

TEST_SUBMODULE(stl, m) {
    // test_vector
    m.def("cast_vector", []() { return std::vector<int>{1}; });
    m.def("load_vector", [](const std::vector<int> &v) { return v.at(0) == 1 && v.at(1) == 2; });
    // `std::vector<bool>` is special because it returns proxy objects instead of references
    m.def("cast_bool_vector", []() { return std::vector<bool>{true, false}; });
    m.def("load_bool_vector", [](const std::vector<bool> &v) {
        return v.at(0) == true && v.at(1) == false;
    });
    // Unnumbered regression (caused by #936): pointers to stl containers aren't castable
    static std::vector<RValueCaster> lvv{2};
    m.def("cast_ptr_vector", []() { return &lvv; });

    // test_array
    m.def("cast_array", []() { return std::array<int, 2> {{1 , 2}}; });
    m.def("load_array", [](const std::array<int, 2> &a) { return a[0] == 1 && a[1] == 2; });

    // test_valarray
    m.def("cast_valarray", []() { return std::valarray<int>{1, 4, 9}; });
    m.def("load_valarray", [](const std::valarray<int>& v) {
        return v.size() == 3 && v[0] == 1 && v[1] == 4 && v[2] == 9;
    });

    // test_map
    m.def("cast_map", []() { return std::map<std::string, std::string>{{"key", "value"}}; });
    m.def("load_map", [](const std::map<std::string, std::string> &map) {
        return map.at("key") == "value" && map.at("key2") == "value2";
    });

    // test_set
    m.def("cast_set", []() { return std::set<std::string>{"key1", "key2"}; });
    m.def("load_set", [](const std::set<std::string> &set) {
        return set.count("key1") && set.count("key2") && set.count("key3");
    });

    // test_recursive_casting
    m.def("cast_rv_vector", []() { return std::vector<RValueCaster>{2}; });
    m.def("cast_rv_array", []() { return std::array<RValueCaster, 3>(); });
    // NB: map and set keys are `const`, so while we technically do move them (as `const Type &&`),
    // casters don't typically do anything with that, which means they fall to the `const Type &`
    // caster.
    m.def("cast_rv_map", []() { return std::unordered_map<std::string, RValueCaster>{{"a", RValueCaster{}}}; });
    m.def("cast_rv_nested", []() {
        std::vector<std::array<std::list<std::unordered_map<std::string, RValueCaster>>, 2>> v;
        v.emplace_back(); // add an array
        v.back()[0].emplace_back(); // add a map to the array
        v.back()[0].back().emplace("b", RValueCaster{});
        v.back()[0].back().emplace("c", RValueCaster{});
        v.back()[1].emplace_back(); // add a map to the array
        v.back()[1].back().emplace("a", RValueCaster{});
        return v;
    });
    static std::array<RValueCaster, 2> lva;
    static std::unordered_map<std::string, RValueCaster> lvm{{"a", RValueCaster{}}, {"b", RValueCaster{}}};
    static std::unordered_map<std::string, std::vector<std::list<std::array<RValueCaster, 2>>>> lvn;
    lvn["a"].emplace_back(); // add a list
    lvn["a"].back().emplace_back(); // add an array
    lvn["a"].emplace_back(); // another list
    lvn["a"].back().emplace_back(); // add an array
    lvn["b"].emplace_back(); // add a list
    lvn["b"].back().emplace_back(); // add an array
    lvn["b"].back().emplace_back(); // add another array
    m.def("cast_lv_vector", []() -> const decltype(lvv) & { return lvv; });
    m.def("cast_lv_array", []() -> const decltype(lva) & { return lva; });
    m.def("cast_lv_map", []() -> const decltype(lvm) & { return lvm; });
    m.def("cast_lv_nested", []() -> const decltype(lvn) & { return lvn; });
    // #853:
    m.def("cast_unique_ptr_vector", []() {
        std::vector<std::unique_ptr<UserType>> v;
        v.emplace_back(new UserType{7});
        v.emplace_back(new UserType{42});
        return v;
    });

    // test_move_out_container
    struct MoveOutContainer {
        struct Value { int value; };
        std::list<Value> move_list() const { return {{0}, {1}, {2}}; }
    };
    py::class_<MoveOutContainer::Value>(m, "MoveOutContainerValue")
        .def_readonly("value", &MoveOutContainer::Value::value);
    py::class_<MoveOutContainer>(m, "MoveOutContainer")
        .def(py::init<>())
        .def_property_readonly("move_list", &MoveOutContainer::move_list);

    // Class that can be move- and copy-constructed, but not assigned
    struct NoAssign {
        int value;

        explicit NoAssign(int value = 0) : value(value) { }
        NoAssign(const NoAssign &) = default;
        NoAssign(NoAssign &&) = default;

        NoAssign &operator=(const NoAssign &) = delete;
        NoAssign &operator=(NoAssign &&) = delete;
    };
    py::class_<NoAssign>(m, "NoAssign", "Class with no C++ assignment operators")
        .def(py::init<>())
        .def(py::init<int>());

    // #528: templated constructor
    // (no python tests: the test here is that this compiles)
    m.def("tpl_ctor_vector", [](std::vector<TplCtorClass> &) {});
    m.def("tpl_ctor_map", [](std::unordered_map<TplCtorClass, TplCtorClass> &) {});
    m.def("tpl_ctor_set", [](std::unordered_set<TplCtorClass> &) {});

    // test_vec_of_reference_wrapper
    // #171: Can't return STL structures containing reference wrapper
    m.def("return_vec_of_reference_wrapper", [](std::reference_wrapper<UserType> p4) {
        static UserType p1{1}, p2{2}, p3{3};
        return std::vector<std::reference_wrapper<UserType>> {
            std::ref(p1), std::ref(p2), std::ref(p3), p4
        };
    });

    // test_stl_pass_by_pointer
    m.def("stl_pass_by_pointer", [](std::vector<int>* v) { return *v; }, "v"_a=nullptr);
}
