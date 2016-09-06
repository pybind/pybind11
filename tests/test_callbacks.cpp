/*
    tests/test_callbacks.cpp -- callbacks

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <pybind11/functional.h>


py::object test_callback1(py::object func) {
    return func();
}

py::tuple test_callback2(py::object func) {
    return func("Hello", 'x', true, 5);
}

std::string test_callback3(const std::function<int(int)> &func) {
    return "func(43) = " + std::to_string(func(43));
}

std::function<int(int)> test_callback4() {
    return [](int i) { return i+1; };
}

py::cpp_function test_callback5() {
    return py::cpp_function([](int i) { return i+1; },
       py::arg("number"));
}

int dummy_function(int i) { return i + 1; }
int dummy_function2(int i, int j) { return i + j; }
std::function<int(int)> roundtrip(std::function<int(int)> f, bool expect_none = false) {
    if (expect_none && f) {
        throw std::runtime_error("Expected None to be converted to empty std::function");
    }
    return f;
}

std::string test_dummy_function(const std::function<int(int)> &f) {
    using fn_type = int (*)(int);
    auto result = f.target<fn_type>();
    if (!result) {
        auto r = f(1);
        return "can't convert to function pointer: eval(1) = " + std::to_string(r);
    } else if (*result == dummy_function) {
        auto r = (*result)(1);
        return "matches dummy_function: eval(1) = " + std::to_string(r);
    } else {
        return "argument does NOT match dummy_function. This should never happen!";
    }
}

struct Payload {
    Payload() {
        print_default_created(this);
    }
    ~Payload() {
        print_destroyed(this);
    }
    Payload(const Payload &) {
        print_copy_created(this);
    }
    Payload(Payload &&) {
        print_move_created(this);
    }
};

/// Something to trigger a conversion error
struct Unregistered {};

test_initializer callbacks([](py::module &m) {
    m.def("test_callback1", &test_callback1);
    m.def("test_callback2", &test_callback2);
    m.def("test_callback3", &test_callback3);
    m.def("test_callback4", &test_callback4);
    m.def("test_callback5", &test_callback5);

    // Test keyword args and generalized unpacking
    m.def("test_tuple_unpacking", [](py::function f) {
        auto t1 = py::make_tuple(2, 3);
        auto t2 = py::make_tuple(5, 6);
        return f("positional", 1, *t1, 4, *t2);
    });

    m.def("test_dict_unpacking", [](py::function f) {
        auto d1 = py::dict("key"_a="value", "a"_a=1);
        auto d2 = py::dict();
        auto d3 = py::dict("b"_a=2);
        return f("positional", 1, **d1, **d2, **d3);
    });

    m.def("test_keyword_args", [](py::function f) {
        return f("x"_a=10, "y"_a=20);
    });

    m.def("test_unpacking_and_keywords1", [](py::function f) {
        auto args = py::make_tuple(2);
        auto kwargs = py::dict("d"_a=4);
        return f(1, *args, "c"_a=3, **kwargs);
    });

    m.def("test_unpacking_and_keywords2", [](py::function f) {
        auto kwargs1 = py::dict("a"_a=1);
        auto kwargs2 = py::dict("c"_a=3, "d"_a=4);
        return f("positional", *py::make_tuple(1), 2, *py::make_tuple(3, 4), 5,
                 "key"_a="value", **kwargs1, "b"_a=2, **kwargs2, "e"_a=5);
    });

    m.def("test_unpacking_error1", [](py::function f) {
        auto kwargs = py::dict("x"_a=3);
        return f("x"_a=1, "y"_a=2, **kwargs); // duplicate ** after keyword
    });

    m.def("test_unpacking_error2", [](py::function f) {
        auto kwargs = py::dict("x"_a=3);
        return f(**kwargs, "x"_a=1); // duplicate keyword after **
    });

    m.def("test_arg_conversion_error1", [](py::function f) {
        f(234, Unregistered(), "kw"_a=567);
    });

    m.def("test_arg_conversion_error2", [](py::function f) {
        f(234, "expected_name"_a=Unregistered(), "kw"_a=567);
    });

    /* Test cleanup of lambda closure */
    m.def("test_cleanup", []() -> std::function<void(void)> {
        Payload p;

        return [p]() {
            /* p should be cleaned up when the returned function is garbage collected */
        };
    });

    /* Test if passing a function pointer from C++ -> Python -> C++ yields the original pointer */
    m.def("dummy_function", &dummy_function);
    m.def("dummy_function2", &dummy_function2);
    m.def("roundtrip", &roundtrip, py::arg("f"), py::arg("expect_none")=false);
    m.def("test_dummy_function", &test_dummy_function);
    // Export the payload constructor statistics for testing purposes:
    m.def("payload_cstats", &ConstructorStats::get<Payload>);
});
