/*
    tests/test_pickling.cpp -- pickle support

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

class Pickleable {
public:
    Pickleable(const std::string &value) : m_value(value) { }
    const std::string &value() const { return m_value; }

    void setExtra1(int extra1) { m_extra1 = extra1; }
    void setExtra2(int extra2) { m_extra2 = extra2; }
    int extra1() const { return m_extra1; }
    int extra2() const { return m_extra2; }
private:
    std::string m_value;
    int m_extra1 = 0;
    int m_extra2 = 0;
};

class PickleableWithDict {
public:
    PickleableWithDict(const std::string &value) : value(value) { }

    std::string value;
    int extra;
};

test_initializer pickling([](py::module &m) {
    py::class_<Pickleable>(m, "Pickleable")
        .def(py::init<std::string>())
        .def("value", &Pickleable::value)
        .def("extra1", &Pickleable::extra1)
        .def("extra2", &Pickleable::extra2)
        .def("setExtra1", &Pickleable::setExtra1)
        .def("setExtra2", &Pickleable::setExtra2)
        // For details on the methods below, refer to
        // http://docs.python.org/3/library/pickle.html#pickling-class-instances
        .def("__getstate__", [](const Pickleable &p) {
            /* Return a tuple that fully encodes the state of the object */
            return py::make_tuple(p.value(), p.extra1(), p.extra2());
        })
        .def("__setstate__", [](Pickleable &p, py::tuple t) {
            if (t.size() != 3)
                throw std::runtime_error("Invalid state!");
            /* Invoke the constructor (need to use in-place version) */
            new (&p) Pickleable(t[0].cast<std::string>());

            /* Assign any additional state */
            p.setExtra1(t[1].cast<int>());
            p.setExtra2(t[2].cast<int>());
        });

#if !defined(PYPY_VERSION)
    py::class_<PickleableWithDict>(m, "PickleableWithDict", py::dynamic_attr())
        .def(py::init<std::string>())
        .def_readwrite("value", &PickleableWithDict::value)
        .def_readwrite("extra", &PickleableWithDict::extra)
        .def("__getstate__", [](py::object self) {
            /* Also include __dict__ in state */
            return py::make_tuple(self.attr("value"), self.attr("extra"), self.attr("__dict__"));
        })
        .def("__setstate__", [](py::object self, py::tuple t) {
            if (t.size() != 3)
                throw std::runtime_error("Invalid state!");
            /* Cast and construct */
            auto& p = self.cast<PickleableWithDict&>();
            new (&p) Pickleable(t[0].cast<std::string>());

            /* Assign C++ state */
            p.extra = t[1].cast<int>();

            /* Assign Python state */
            self.attr("__dict__") = t[2];
        });
#endif
});
