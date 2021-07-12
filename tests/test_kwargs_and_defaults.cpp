/*
    tests/test_kwargs_and_defaults.cpp -- keyword arguments and default values

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"
#include <pybind11/stl.h>

#include <utility>

TEST_SUBMODULE(kwargs_and_defaults, m) {
    auto kw_func = [](int x, int y) { return "x=" + std::to_string(x) + ", y=" + std::to_string(y); };

    // test_named_arguments
    m.def("kw_func0", kw_func);
    m.def("kw_func1", kw_func, py::arg("x"), py::arg("y"));
    m.def("kw_func2", kw_func, py::arg("x") = 100, py::arg("y") = 200);
    m.def("kw_func3", [](const char *) { }, py::arg("data") = std::string("Hello world!"));

    /* A fancier default argument */
    std::vector<int> list{{13, 17}};
    m.def("kw_func4", [](const std::vector<int> &entries) {
        std::string ret = "{";
        for (int i : entries)
            ret += std::to_string(i) + " ";
        ret.back() = '}';
        return ret;
    }, py::arg("myList") = list);

    m.def("kw_func_udl", kw_func, "x"_a, "y"_a=300);
    m.def("kw_func_udl_z", kw_func, "x"_a, "y"_a=0);

    // test_args_and_kwargs
    m.def("args_function", [](py::args args) -> py::tuple {
        return std::move(args);
    });
    m.def("args_kwargs_function", [](const py::args &args, const py::kwargs &kwargs) {
        return py::make_tuple(args, kwargs);
    });

    // test_mixed_args_and_kwargs
    m.def("mixed_plus_args",
          [](int i, double j, const py::args &args) { return py::make_tuple(i, j, args); });
    m.def("mixed_plus_kwargs",
          [](int i, double j, const py::kwargs &kwargs) { return py::make_tuple(i, j, kwargs); });
    auto mixed_plus_both = [](int i, double j, const py::args &args, const py::kwargs &kwargs) {
        return py::make_tuple(i, j, args, kwargs);
    };
    m.def("mixed_plus_args_kwargs", mixed_plus_both);

    m.def("mixed_plus_args_kwargs_defaults", mixed_plus_both,
            py::arg("i") = 1, py::arg("j") = 3.14159);

    // test_args_refcount
    // PyPy needs a garbage collection to get the reference count values to match CPython's behaviour
    #ifdef PYPY_VERSION
    #define GC_IF_NEEDED ConstructorStats::gc()
    #else
    #define GC_IF_NEEDED
    #endif
    m.def("arg_refcount_h", [](py::handle h) { GC_IF_NEEDED; return h.ref_count(); });
    m.def("arg_refcount_h", [](py::handle h, py::handle, py::handle) { GC_IF_NEEDED; return h.ref_count(); });
    m.def("arg_refcount_o", [](const py::object &o) {
        GC_IF_NEEDED;
        return o.ref_count();
    });
    m.def("args_refcount", [](py::args a) {
        GC_IF_NEEDED;
        py::tuple t(a.size());
        for (size_t i = 0; i < a.size(); i++)
            // Use raw Python API here to avoid an extra, intermediate incref on the tuple item:
            t[i] = (int) Py_REFCNT(PyTuple_GET_ITEM(a.ptr(), static_cast<py::ssize_t>(i)));
        return t;
    });
    m.def("mixed_args_refcount", [](const py::object &o, py::args a) {
        GC_IF_NEEDED;
        py::tuple t(a.size() + 1);
        t[0] = o.ref_count();
        for (size_t i = 0; i < a.size(); i++)
            // Use raw Python API here to avoid an extra, intermediate incref on the tuple item:
            t[i + 1] = (int) Py_REFCNT(PyTuple_GET_ITEM(a.ptr(), static_cast<py::ssize_t>(i)));
        return t;
    });

    // pybind11 won't allow these to be bound: args and kwargs, if present, must be at the end.
    // Uncomment these to test that the static_assert is indeed working:
//    m.def("bad_args1", [](py::args, int) {});
//    m.def("bad_args2", [](py::kwargs, int) {});
//    m.def("bad_args3", [](py::kwargs, py::args) {});
//    m.def("bad_args4", [](py::args, int, py::kwargs) {});
//    m.def("bad_args5", [](py::args, py::kwargs, int) {});
//    m.def("bad_args6", [](py::args, py::args) {});
//    m.def("bad_args7", [](py::kwargs, py::kwargs) {});

    // test_keyword_only_args
    m.def("kw_only_all", [](int i, int j) { return py::make_tuple(i, j); },
            py::kw_only(), py::arg("i"), py::arg("j"));
    m.def("kw_only_some", [](int i, int j, int k) { return py::make_tuple(i, j, k); },
            py::arg(), py::kw_only(), py::arg("j"), py::arg("k"));
    m.def("kw_only_with_defaults", [](int i, int j, int k, int z) { return py::make_tuple(i, j, k, z); },
            py::arg() = 3, "j"_a = 4, py::kw_only(), "k"_a = 5, "z"_a);
    m.def("kw_only_mixed", [](int i, int j) { return py::make_tuple(i, j); },
            "i"_a, py::kw_only(), "j"_a);
    m.def(
        "kw_only_plus_more",
        [](int i, int j, int k, const py::kwargs &kwargs) {
            return py::make_tuple(i, j, k, kwargs);
        },
        py::arg() /* positional */,
        py::arg("j") = -1 /* both */,
        py::kw_only(),
        py::arg("k") /* kw-only */);

    m.def("register_invalid_kw_only", [](py::module_ m) {
        m.def("bad_kw_only", [](int i, int j) { return py::make_tuple(i, j); },
                py::kw_only(), py::arg() /* invalid unnamed argument */, "j"_a);
    });

    // test_positional_only_args
    m.def("pos_only_all", [](int i, int j) { return py::make_tuple(i, j); },
            py::arg("i"), py::arg("j"), py::pos_only());
    m.def("pos_only_mix", [](int i, int j) { return py::make_tuple(i, j); },
            py::arg("i"), py::pos_only(), py::arg("j"));
    m.def("pos_kw_only_mix", [](int i, int j, int k) { return py::make_tuple(i, j, k); },
            py::arg("i"), py::pos_only(), py::arg("j"), py::kw_only(), py::arg("k"));
    m.def("pos_only_def_mix", [](int i, int j, int k) { return py::make_tuple(i, j, k); },
            py::arg("i"), py::arg("j") = 2, py::pos_only(), py::arg("k") = 3);


    // These should fail to compile:
    // argument annotations are required when using kw_only
//    m.def("bad_kw_only1", [](int) {}, py::kw_only());
    // can't specify both `py::kw_only` and a `py::args` argument
//    m.def("bad_kw_only2", [](int i, py::args) {}, py::kw_only(), "i"_a);

    // test_function_signatures (along with most of the above)
    struct KWClass { void foo(int, float) {} };
    py::class_<KWClass>(m, "KWClass")
        .def("foo0", &KWClass::foo)
        .def("foo1", &KWClass::foo, "x"_a, "y"_a);

    // Make sure a class (not an instance) can be used as a default argument.
    // The return value doesn't matter, only that the module is importable.
    m.def(
        "class_default_argument",
        [](py::object a) { return py::repr(std::move(a)); },
        "a"_a = py::module_::import("decimal").attr("Decimal"));
}
