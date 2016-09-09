/*
    tests/test_python_types.cpp -- singleton design pattern, static functions and
    variables, passing and interacting with Python types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <pybind11/stl.h>

#ifdef _WIN32
#  include <io.h>
#  include <fcntl.h>
#endif

class ExamplePythonTypes {
public:
    static ExamplePythonTypes *new_instance() {
        auto *ptr = new ExamplePythonTypes();
        print_created(ptr, "via new_instance");
        return ptr;
    }
    ~ExamplePythonTypes() { print_destroyed(this); }

    /* Create and return a Python dictionary */
    py::dict get_dict() {
        py::dict dict;
        dict[py::str("key")] = py::str("value");
        return dict;
    }

    /* Create and return a Python set */
    py::set get_set() {
        py::set set;
        set.add(py::str("key1"));
        set.add(py::str("key2"));
        return set;
    }

    /* Create and return a C++ dictionary */
    std::map<std::string, std::string> get_dict_2() {
        std::map<std::string, std::string> result;
        result["key"] = "value";
        return result;
    }

    /* Create and return a C++ set */
    std::set<std::string> get_set_2() {
        std::set<std::string> result;
        result.insert("key1");
        result.insert("key2");
        return result;
    }

    /* Create, manipulate, and return a Python list */
    py::list get_list() {
        py::list list;
        list.append(py::str("value"));
        py::print("Entry at position 0:", py::object(list[0]));
        list[0] = py::str("overwritten");
        return list;
    }

    /* C++ STL data types are automatically casted */
    std::vector<std::wstring> get_list_2() {
        std::vector<std::wstring> list;
        list.push_back(L"value");
        return list;
    }

    /* C++ STL data types are automatically casted */
    std::array<std::string, 2> get_array() {
        return std::array<std::string, 2> {{ "array entry 1" , "array entry 2"}};
    }

    /* Easily iterate over a dictionary using a C++11 range-based for loop */
    void print_dict(py::dict dict) {
        for (auto item : dict)
            py::print("key: {}, value={}"_s.format(item.first, item.second));
    }

    /* Easily iterate over a set using a C++11 range-based for loop */
    void print_set(py::set set) {
        for (auto item : set)
            py::print("key:", item);
    }

    /* Easily iterate over a list using a C++11 range-based for loop */
    void print_list(py::list list) {
        int index = 0;
        for (auto item : list)
            py::print("list item {}: {}"_s.format(index++, item));
    }

    /* STL data types (such as maps) are automatically casted from Python */
    void print_dict_2(const std::map<std::string, std::string> &dict) {
        for (auto item : dict)
            py::print("key: {}, value={}"_s.format(item.first, item.second));
    }

    /* STL data types (such as sets) are automatically casted from Python */
    void print_set_2(const std::set<std::string> &set) {
        for (auto item : set)
            py::print("key:", item);
    }

    /* STL data types (such as vectors) are automatically casted from Python */
    void print_list_2(std::vector<std::wstring> &list) {
        int index = 0;
        for (auto item : list)
            py::print("list item {}: {}"_s.format(index++, item));
    }

    /* pybind automatically translates between C++11 and Python tuples */
    std::pair<std::string, bool> pair_passthrough(std::pair<bool, std::string> input) {
        return std::make_pair(input.second, input.first);
    }

    /* pybind automatically translates between C++11 and Python tuples */
    std::tuple<int, std::string, bool> tuple_passthrough(std::tuple<bool, std::string, int> input) {
        return std::make_tuple(std::get<2>(input), std::get<1>(input), std::get<0>(input));
    }

    /* STL data types (such as arrays) are automatically casted from Python */
    void print_array(std::array<std::string, 2> &array) {
        int index = 0;
        for (auto item : array)
            py::print("array item {}: {}"_s.format(index++, item));
    }

    void throw_exception() {
        throw std::runtime_error("This exception was intentionally thrown.");
    }

    py::bytes get_bytes_from_string() {
        return (py::bytes) std::string("foo");
    }

    py::bytes get_bytes_from_str() {
        return (py::bytes) py::str("bar", 3);
    }

    py::str get_str_from_string() {
        return (py::str) std::string("baz");
    }

    py::str get_str_from_bytes() {
        return (py::str) py::bytes("boo", 3);
    }

    void test_print(const py::object& obj) {
        py::print(obj.str());
        py::print(obj.repr());
    }

    static int value;
    static const int value2;
};

int ExamplePythonTypes::value = 0;
const int ExamplePythonTypes::value2 = 5;

test_initializer python_types([](py::module &m) {
    /* No constructor is explicitly defined below. An exception is raised when
       trying to construct it directly from Python */
    py::class_<ExamplePythonTypes>(m, "ExamplePythonTypes", "Example 2 documentation")
        .def("get_dict", &ExamplePythonTypes::get_dict, "Return a Python dictionary")
        .def("get_dict_2", &ExamplePythonTypes::get_dict_2, "Return a C++ dictionary")
        .def("get_list", &ExamplePythonTypes::get_list, "Return a Python list")
        .def("get_list_2", &ExamplePythonTypes::get_list_2, "Return a C++ list")
        .def("get_set", &ExamplePythonTypes::get_set, "Return a Python set")
        .def("get_set2", &ExamplePythonTypes::get_set_2, "Return a C++ set")
        .def("get_array", &ExamplePythonTypes::get_array, "Return a C++ array")
        .def("print_dict", &ExamplePythonTypes::print_dict, "Print entries of a Python dictionary")
        .def("print_dict_2", &ExamplePythonTypes::print_dict_2, "Print entries of a C++ dictionary")
        .def("print_set", &ExamplePythonTypes::print_set, "Print entries of a Python set")
        .def("print_set_2", &ExamplePythonTypes::print_set_2, "Print entries of a C++ set")
        .def("print_list", &ExamplePythonTypes::print_list, "Print entries of a Python list")
        .def("print_list_2", &ExamplePythonTypes::print_list_2, "Print entries of a C++ list")
        .def("print_array", &ExamplePythonTypes::print_array, "Print entries of a C++ array")
        .def("pair_passthrough", &ExamplePythonTypes::pair_passthrough, "Return a pair in reversed order")
        .def("tuple_passthrough", &ExamplePythonTypes::tuple_passthrough, "Return a triple in reversed order")
        .def("throw_exception", &ExamplePythonTypes::throw_exception, "Throw an exception")
        .def("get_bytes_from_string", &ExamplePythonTypes::get_bytes_from_string, "py::bytes from std::string")
        .def("get_bytes_from_str", &ExamplePythonTypes::get_bytes_from_str, "py::bytes from py::str")
        .def("get_str_from_string", &ExamplePythonTypes::get_str_from_string, "py::str from std::string")
        .def("get_str_from_bytes", &ExamplePythonTypes::get_str_from_bytes, "py::str from py::bytes")
        .def("test_print", &ExamplePythonTypes::test_print, "test the print function")
        .def_static("new_instance", &ExamplePythonTypes::new_instance, "Return an instance")
        .def_readwrite_static("value", &ExamplePythonTypes::value, "Static value member")
        .def_readonly_static("value2", &ExamplePythonTypes::value2, "Static value member (readonly)")
        ;

    m.def("test_print_function", []() {
        py::print("Hello, World!");
        py::print(1, 2.0, "three", true, std::string("-- multiple args"));
        auto args = py::make_tuple("and", "a", "custom", "separator");
        py::print("*args", *args, "sep"_a="-");
        py::print("no new line here", "end"_a=" -- ");
        py::print("next print");

        auto py_stderr = py::module::import("sys").attr("stderr").cast<py::object>();
        py::print("this goes to stderr", "file"_a=py_stderr);

        py::print("flush", "flush"_a=true);

        py::print("{a} + {b} = {c}"_s.format("a"_a="py::print", "b"_a="str.format", "c"_a="this"));
    });

    m.def("test_str_format", []() {
        auto s1 = "{} + {} = {}"_s.format(1, 2, 3);
        auto s2 = "{a} + {b} = {c}"_s.format("a"_a=1, "b"_a=2, "c"_a=3);
        return py::make_tuple(s1, s2);
    });

    m.def("test_dict_keyword_constructor", []() {
        auto d1 = py::dict("x"_a=1, "y"_a=2);
        auto d2 = py::dict("z"_a=3, **d1);
        return d2;
    });
});
