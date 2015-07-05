/*
    example/example6.cpp -- Example 6: supporting Pythons' sequence
    protocol, iterators, etc.

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"
#include <pybind/operators.h>

class Sequence {
public:
    Sequence(size_t size) : m_size(size) {
        std::cout << "Value constructor: Creating a sequence with " << m_size << " entries" << std::endl;
        m_data = new float[size];
        memset(m_data, 0, sizeof(float) * size);
    }

    Sequence(const std::vector<float> &value) : m_size(value.size()) {
        std::cout << "Value constructor: Creating a sequence with " << m_size << " entries" << std::endl;
        m_data = new float[m_size];
        memcpy(m_data, &value[0], sizeof(float) * m_size);
    }

    Sequence(const Sequence &s) : m_size(s.m_size) {
        std::cout << "Copy constructor: Creating a sequence with " << m_size << " entries" << std::endl;
        m_data = new float[m_size];
        memcpy(m_data, s.m_data, sizeof(float)*m_size);
    }

    Sequence(Sequence &&s) : m_size(s.m_size), m_data(s.m_data) {
        std::cout << "Move constructor: Creating a sequence with " << m_size << " entries" << std::endl;
        s.m_size = 0;
        s.m_data = nullptr;
    }

    ~Sequence() {
        std::cout << "Freeing a sequence with " << m_size << " entries" << std::endl;
        delete[] m_data;
    }

    Sequence &operator=(const Sequence &s) {
        std::cout << "Assignment operator: Creating a sequence with " << s.m_size << " entries" << std::endl;
        delete[] m_data;
        m_size = s.m_size;
        m_data = new float[m_size];
        memcpy(m_data, s.m_data, sizeof(float)*m_size);
        return *this;
    }

    Sequence &operator=(Sequence &&s) {
        std::cout << "Move assignment operator: Creating a sequence with " << s.m_size << " entries" << std::endl;
        if (&s != this) {
            delete[] m_data;
            m_size = s.m_size;
            m_data = s.m_data;
            s.m_size = 0;
            s.m_data = nullptr;
        }
        return *this;
    }

    bool operator==(const Sequence &s) const {
        if (m_size != s.size())
            return false;
        for (size_t i=0; i<m_size; ++i)
            if (m_data[i] != s[i])
                return false;
        return true;
    }

    bool operator!=(const Sequence &s) const {
        return !operator==(s);
    }

    float operator[](size_t index) const {
        return m_data[index];
    }

    float &operator[](size_t index) {
        return m_data[index];
    }

    bool contains(float v) const {
        for (size_t i=0; i<m_size; ++i)
            if (v == m_data[i])
                return true;
        return false;
    }

    Sequence reversed() const {
        Sequence result(m_size);
        for (size_t i=0; i<m_size; ++i)
            result[m_size-i-1] = m_data[i];
        return result;
    }

    size_t size() const { return m_size; }

private:
    size_t m_size;
    float *m_data;
};

namespace {
    // Special iterator data structure for python
    struct PySequenceIterator {
        PySequenceIterator(const Sequence &seq, py::object ref) : seq(seq), ref(ref) { }

        float next() {
            if (index == seq.size())
                throw py::stop_iteration();
            return seq[index++];
        }

        const Sequence &seq;
        py::object ref; // keep a reference
        size_t index = 0;
    };
};

void init_ex6(py::module &m) {
    py::class_<Sequence> seq(m, "Sequence");

    seq.def(py::init<size_t>())
       .def(py::init<const std::vector<float>&>())
       /// Bare bones interface
       .def("__getitem__", [](const Sequence &s, size_t i) {
            if (i >= s.size())
                throw py::index_error();
            return s[i];
        })
       .def("__setitem__", [](Sequence &s, size_t i, float v) {
            if (i >= s.size())
                throw py::index_error();
            s[i] = v;
        })
       .def("__len__", &Sequence::size)
       /// Optional sequence protocol operations
       .def("__iter__", [](py::object s) { return PySequenceIterator(s.cast<const Sequence &>(), s); })
       .def("__contains__", [](const Sequence &s, float v) { return s.contains(v); })
       .def("__reversed__", [](const Sequence &s) -> Sequence { return s.reversed(); })
       /// Slicing protocol (optional)
       .def("__getitem__", [](const Sequence &s, py::slice slice) -> Sequence* {
            py::ssize_t start, stop, step, slicelength;
            if (!slice.compute(s.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            Sequence *seq = new Sequence(slicelength);
            for (int i=0; i<slicelength; ++i) {
                (*seq)[i] = s[start]; start += step;
            }
            return seq;
        })
       .def("__setitem__", [](Sequence &s, py::slice slice, const Sequence &value) {
            py::ssize_t start, stop, step, slicelength;
            if (!slice.compute(s.size(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();
            if ((size_t) slicelength != value.size())
                throw std::runtime_error("Left and right hand size of slice assignment have different sizes!");
            for (int i=0; i<slicelength; ++i) {
                s[start] = value[i]; start += step;
            }
        })
       /// Comparisons
       .def(py::self == py::self)
       .def(py::self != py::self);
       // Could also define py::self + py::self for concatenation, etc.

    py::class_<PySequenceIterator>(seq, "Iterator")
        .def("__iter__", [](PySequenceIterator &it) -> PySequenceIterator& { return it; })
        .def("__next__", &PySequenceIterator::next);
}
