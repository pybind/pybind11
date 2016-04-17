/*
    example/example2.cpp2 -- singleton design pattern, static functions and
    variables, passing and interacting with Python types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"
#include <pybind11/stl.h>

#ifdef _WIN32
#  include <io.h>
#  include <fcntl.h>
#endif

class Example2 {
public:
    static Example2 *new_instance() {
        return new Example2();
    }
    ~Example2() {
        std::cout << "Destructing Example2" << std::endl;
    }

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
        cout << "Entry at positon 0: " << py::object(list[0]) << endl;
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
            std::cout << "key: " << item.first << ", value=" << item.second << std::endl;
    }

    /* Easily iterate over a set using a C++11 range-based for loop */
    void print_set(py::set set) {
        for (auto item : set)
            std::cout << "key: " << item << std::endl;
    }

    /* Easily iterate over a list using a C++11 range-based for loop */
    void print_list(py::list list) {
        int index = 0;
        for (auto item : list)
            std::cout << "list item " << index++ << ": " << item << std::endl;
    }

    /* STL data types (such as maps) are automatically casted from Python */
    void print_dict_2(const std::map<std::string, std::string> &dict) {
        for (auto item : dict)
            std::cout << "key: " << item.first << ", value=" << item.second << std::endl;
    }

    /* STL data types (such as sets) are automatically casted from Python */
    void print_set_2(const std::set<std::string> &set) {
        for (auto item : set)
            std::cout << "key: " << item << std::endl;
    }

    /* STL data types (such as vectors) are automatically casted from Python */
    void print_list_2(std::vector<std::wstring> &list) {
#ifdef _WIN32 /* Can't easily mix cout and wcout on Windows */
        _setmode(_fileno(stdout), _O_TEXT);
#endif
        int index = 0;
        for (auto item : list)
            std::wcout << L"list item " << index++ << L": " << item << std::endl;
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
            std::cout << "array item " << index++ << ": " << item << std::endl;
    }

    void throw_exception() {
        throw std::runtime_error("This exception was intentionally thrown.");
    }

    static int value;
    static const int value2;
};

int Example2::value = 0;
const int Example2::value2 = 5;

void init_ex2(py::module &m) {
    /* No constructor is explicitly defined below. An exception is raised when
       trying to construct it directly from Python */
    py::class_<Example2>(m, "Example2", "Example 2 documentation")
        .def("get_dict", &Example2::get_dict, "Return a Python dictionary")
        .def("get_dict_2", &Example2::get_dict_2, "Return a C++ dictionary")
        .def("get_list", &Example2::get_list, "Return a Python list")
        .def("get_list_2", &Example2::get_list_2, "Return a C++ list")
        .def("get_set", &Example2::get_set, "Return a Python set")
        .def("get_set2", &Example2::get_set, "Return a C++ set")
        .def("get_array", &Example2::get_array, "Return a C++ array")
        .def("print_dict", &Example2::print_dict, "Print entries of a Python dictionary")
        .def("print_dict_2", &Example2::print_dict_2, "Print entries of a C++ dictionary")
        .def("print_set", &Example2::print_set, "Print entries of a Python set")
        .def("print_set_2", &Example2::print_set_2, "Print entries of a C++ set")
        .def("print_list", &Example2::print_list, "Print entries of a Python list")
        .def("print_list_2", &Example2::print_list_2, "Print entries of a C++ list")
        .def("print_array", &Example2::print_array, "Print entries of a C++ array")
        .def("pair_passthrough", &Example2::pair_passthrough, "Return a pair in reversed order")
        .def("tuple_passthrough", &Example2::tuple_passthrough, "Return a triple in reversed order")
        .def("throw_exception", &Example2::throw_exception, "Throw an exception")
        .def_static("new_instance", &Example2::new_instance, "Return an instance")
        .def_readwrite_static("value", &Example2::value, "Static value member")
        .def_readonly_static("value2", &Example2::value2, "Static value member (readonly)");
}
