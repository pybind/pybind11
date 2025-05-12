#include "pybind11_tests.h"

#include <Python.h>
#include <memory>
#include <vector>

class VectorOwns4PythonObjects {
public:
    void append(const py::object &obj) {
        if (size() >= 4) {
            throw std::out_of_range("Index out of range");
        }
        vec.emplace_back(obj);
    }

    void set_item(py::ssize_t i, const py::object &obj) {
        if (!(i >= 0 && i < size())) {
            throw std::out_of_range("Index out of range");
        }
        vec[py::size_t(i)] = obj;
    }

    py::object get_item(py::ssize_t i) const {
        if (!(i >= 0 && i < size())) {
            throw std::out_of_range("Index out of range");
        }
        return vec[py::size_t(i)];
    }

    py::ssize_t size() const { return py::ssize_t_cast(vec.size()); }

    bool is_empty() const { return vec.empty(); }

    void sanity_check() const {
        auto current_size = size();
        if (current_size < 0 || current_size > 4) {
            throw std::out_of_range("Invalid size");
        }
    }

    static int tp_traverse(PyObject *self_base, visitproc visit, void *arg) {
#if PY_VERSION_HEX >= 0x03090000 // Python 3.9
        Py_VISIT(Py_TYPE(self_base));
#endif
        auto *const instance = reinterpret_cast<py::detail::instance *>(self_base);
        if (!instance->get_value_and_holder().holder_constructed()) {
            // The holder has not been constructed yet. Skip the traversal to avoid segmentation
            // faults.
            return 0;
        }
        auto &self = py::cast<VectorOwns4PythonObjects &>(py::handle{self_base});
        for (const auto &obj : self.vec) {
            Py_VISIT(obj.ptr());
        }
        return 0;
    }

private:
    std::vector<py::object> vec{};
};

TEST_SUBMODULE(invalid_holder_access, m) {
    m.doc() = "Test invalid holder access";

#if defined(PYBIND11_CPP14)
    m.def("create_vector", []() -> std::unique_ptr<VectorOwns4PythonObjects> {
        auto vec = std::make_unique<VectorOwns4PythonObjects>();
        vec->append(py::none());
        vec->append(py::int_(1));
        vec->append(py::str("test"));
        vec->append(py::tuple());
        return vec;
    });
#endif

    py::class_<VectorOwns4PythonObjects>(
        m,
        "VectorOwns4PythonObjects",
        py::custom_type_setup([](PyHeapTypeObject *heap_type) -> void {
            auto *const type = &heap_type->ht_type;
            type->tp_flags |= Py_TPFLAGS_HAVE_GC;
            type->tp_traverse = &VectorOwns4PythonObjects::tp_traverse;
        }))
        .def("append", &VectorOwns4PythonObjects::append, py::arg("obj"))
        .def("set_item", &VectorOwns4PythonObjects::set_item, py::arg("i"), py::arg("obj"))
        .def("get_item", &VectorOwns4PythonObjects::get_item, py::arg("i"))
        .def("size", &VectorOwns4PythonObjects::size)
        .def("is_empty", &VectorOwns4PythonObjects::is_empty)
        .def("__setitem__", &VectorOwns4PythonObjects::set_item, py::arg("i"), py::arg("obj"))
        .def("__getitem__", &VectorOwns4PythonObjects::get_item, py::arg("i"))
        .def("__len__", &VectorOwns4PythonObjects::size)
        .def("sanity_check", &VectorOwns4PythonObjects::sanity_check);
}
