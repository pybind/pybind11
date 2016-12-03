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
        set.add("key2");
        set.add(std::string("key3"));
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
        list.append("value");
        py::print("Entry at position 0:", list[0]);
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

    std::valarray<int> get_valarray() {
        return std::valarray<int>({ 1, 4, 9 });
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

    void print_valarray(std::valarray<int> &varray) {
        int index = 0;
        for (auto item : varray)
            py::print("valarray item {}: {}"_s.format(index++, item));
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
        py::print(py::str(obj));
        py::print(py::repr(obj));
    }

    static int value;
    static const int value2;
};

int ExamplePythonTypes::value = 0;
const int ExamplePythonTypes::value2 = 5;

struct MoveOutContainer {
    struct Value { int value; };

    std::list<Value> move_list() const { return {{0}, {1}, {2}}; }
};


test_initializer python_types([](py::module &m) {
    /* No constructor is explicitly defined below. An exception is raised when
       trying to construct it directly from Python */
    py::class_<ExamplePythonTypes>(m, "ExamplePythonTypes", "Example 2 documentation", py::metaclass())
        .def("get_dict", &ExamplePythonTypes::get_dict, "Return a Python dictionary")
        .def("get_dict_2", &ExamplePythonTypes::get_dict_2, "Return a C++ dictionary")
        .def("get_list", &ExamplePythonTypes::get_list, "Return a Python list")
        .def("get_list_2", &ExamplePythonTypes::get_list_2, "Return a C++ list")
        .def("get_set", &ExamplePythonTypes::get_set, "Return a Python set")
        .def("get_set2", &ExamplePythonTypes::get_set_2, "Return a C++ set")
        .def("get_array", &ExamplePythonTypes::get_array, "Return a C++ array")
        .def("get_valarray", &ExamplePythonTypes::get_valarray, "Return a C++ valarray")
        .def("print_dict", &ExamplePythonTypes::print_dict, "Print entries of a Python dictionary")
        .def("print_dict_2", &ExamplePythonTypes::print_dict_2, "Print entries of a C++ dictionary")
        .def("print_set", &ExamplePythonTypes::print_set, "Print entries of a Python set")
        .def("print_set_2", &ExamplePythonTypes::print_set_2, "Print entries of a C++ set")
        .def("print_list", &ExamplePythonTypes::print_list, "Print entries of a Python list")
        .def("print_list_2", &ExamplePythonTypes::print_list_2, "Print entries of a C++ list")
        .def("print_array", &ExamplePythonTypes::print_array, "Print entries of a C++ array")
        .def("print_valarray", &ExamplePythonTypes::print_valarray, "Print entries of a C++ valarray")
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
        .def_readonly_static("value2", &ExamplePythonTypes::value2, "Static value member (readonly)");

    m.def("test_print_function", []() {
        py::print("Hello, World!");
        py::print(1, 2.0, "three", true, std::string("-- multiple args"));
        auto args = py::make_tuple("and", "a", "custom", "separator");
        py::print("*args", *args, "sep"_a="-");
        py::print("no new line here", "end"_a=" -- ");
        py::print("next print");

        auto py_stderr = py::module::import("sys").attr("stderr");
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

    m.def("test_accessor_api", [](py::object o) {
        auto d = py::dict();

        d["basic_attr"] = o.attr("basic_attr");

        auto l = py::list();
        for (const auto &item : o.attr("begin_end")) {
            l.append(item);
        }
        d["begin_end"] = l;

        d["operator[object]"] = o.attr("d")["operator[object]"_s];
        d["operator[char *]"] = o.attr("d")["operator[char *]"];

        d["attr(object)"] = o.attr("sub").attr("attr_obj");
        d["attr(char *)"] = o.attr("sub").attr("attr_char");
        try {
            o.attr("sub").attr("missing").ptr();
        } catch (const py::error_already_set &) {
            d["missing_attr_ptr"] = "raised"_s;
        }
        try {
            o.attr("missing").attr("doesn't matter");
        } catch (const py::error_already_set &) {
            d["missing_attr_chain"] = "raised"_s;
        }

        d["is_none"] = o.attr("basic_attr").is_none();

        d["operator()"] = o.attr("func")(1);
        d["operator*"] = o.attr("func")(*o.attr("begin_end"));

        return d;
    });

    m.def("test_tuple_accessor", [](py::tuple existing_t) {
        try {
            existing_t[0] = 1;
        } catch (const py::error_already_set &) {
            // --> Python system error
            // Only new tuples (refcount == 1) are mutable
            auto new_t = py::tuple(3);
            for (size_t i = 0; i < new_t.size(); ++i) {
                new_t[i] = i;
            }
            return new_t;
        }
        return py::tuple();
    });

    m.def("test_accessor_assignment", []() {
        auto l = py::list(1);
        l[0] = 0;

        auto d = py::dict();
        d["get"] = l[0];
        auto var = l[0];
        d["deferred_get"] = var;
        l[0] = 1;
        d["set"] = l[0];
        var = 99; // this assignment should not overwrite l[0]
        d["deferred_set"] = l[0];
        d["var"] = var;

        return d;
    });

    bool has_optional = false, has_exp_optional = false;
#ifdef PYBIND11_HAS_OPTIONAL
    has_optional = true;
    using opt_int = std::optional<int>;
    m.def("double_or_zero", [](const opt_int& x) -> int {
        return x.value_or(0) * 2;
    });
    m.def("half_or_none", [](int x) -> opt_int {
        return x ? opt_int(x / 2) : opt_int();
    });
    m.def("test_nullopt", [](opt_int x) {
        return x.value_or(42);
    }, py::arg_v("x", std::nullopt, "None"));
#endif

#ifdef PYBIND11_HAS_EXP_OPTIONAL
    has_exp_optional = true;
    using exp_opt_int = std::experimental::optional<int>;
    m.def("double_or_zero_exp", [](const exp_opt_int& x) -> int {
        return x.value_or(0) * 2;
    });
    m.def("half_or_none_exp", [](int x) -> exp_opt_int {
        return x ? exp_opt_int(x / 2) : exp_opt_int();
    });
    m.def("test_nullopt_exp", [](exp_opt_int x) {
        return x.value_or(42);
    }, py::arg_v("x", std::experimental::nullopt, "None"));
#endif

    m.attr("has_optional") = has_optional;
    m.attr("has_exp_optional") = has_exp_optional;

    m.def("test_default_constructors", []() {
        return py::dict(
            "str"_a=py::str(),
            "bool"_a=py::bool_(),
            "int"_a=py::int_(),
            "float"_a=py::float_(),
            "tuple"_a=py::tuple(),
            "list"_a=py::list(),
            "dict"_a=py::dict(),
            "set"_a=py::set()
        );
    });

    m.def("test_converting_constructors", [](py::dict d) {
        return py::dict(
            "str"_a=py::str(d["str"]),
            "bool"_a=py::bool_(d["bool"]),
            "int"_a=py::int_(d["int"]),
            "float"_a=py::float_(d["float"]),
            "tuple"_a=py::tuple(d["tuple"]),
            "list"_a=py::list(d["list"]),
            "dict"_a=py::dict(d["dict"]),
            "set"_a=py::set(d["set"]),
            "memoryview"_a=py::memoryview(d["memoryview"])
        );
    });

    m.def("test_cast_functions", [](py::dict d) {
        // When converting between Python types, obj.cast<T>() should be the same as T(obj)
        return py::dict(
            "str"_a=d["str"].cast<py::str>(),
            "bool"_a=d["bool"].cast<py::bool_>(),
            "int"_a=d["int"].cast<py::int_>(),
            "float"_a=d["float"].cast<py::float_>(),
            "tuple"_a=d["tuple"].cast<py::tuple>(),
            "list"_a=d["list"].cast<py::list>(),
            "dict"_a=d["dict"].cast<py::dict>(),
            "set"_a=d["set"].cast<py::set>(),
            "memoryview"_a=d["memoryview"].cast<py::memoryview>()
        );
    });

    py::class_<MoveOutContainer::Value>(m, "MoveOutContainerValue")
        .def_readonly("value", &MoveOutContainer::Value::value);

    py::class_<MoveOutContainer>(m, "MoveOutContainer")
        .def(py::init<>())
        .def_property_readonly("move_list", &MoveOutContainer::move_list);

    m.def("get_implicit_casting", []() {
        py::dict d;
        d["char*_i1"] = "abc";
        const char *c2 = "abc";
        d["char*_i2"] = c2;
        d["char*_e"] = py::cast(c2);
        d["char*_p"] = py::str(c2);

        d["int_i1"] = 42;
        int i = 42;
        d["int_i2"] = i;
        i++;
        d["int_e"] = py::cast(i);
        i++;
        d["int_p"] = py::int_(i);

        d["str_i1"] = std::string("str");
        std::string s2("str1");
        d["str_i2"] = s2;
        s2[3] = '2';
        d["str_e"] = py::cast(s2);
        s2[3] = '3';
        d["str_p"] = py::str(s2);

        py::list l(2);
        l[0] = 3;
        l[1] = py::cast(6);
        l.append(9);
        l.append(py::cast(12));
        l.append(py::int_(15));

        return py::dict(
            "d"_a=d,
            "l"_a=l
        );
    });
});
