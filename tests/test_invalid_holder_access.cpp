#include "pybind11_tests.h"

#include <Python.h>
#include <memory>
#include <vector>

class VecOwnsObjs {
public:
    void append(const py::object &obj) { vec.emplace_back(obj); }

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

    static int tp_traverse(PyObject *self_base, visitproc visit, void *arg) {
        // https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_traverse
#if PY_VERSION_HEX >= 0x03090000 // Python 3.9
        Py_VISIT(Py_TYPE(self_base));
#endif

        if (should_check_holder_initialization && !py::detail::is_holder_constructed(self_base)) {
            // The holder has not been constructed yet. Skip the traversal to avoid
            // segmentation faults.
            return 0;
        }

        // The actual logic of the tp_traverse function goes here.
        auto &self = py::cast<VecOwnsObjs &>(py::handle{self_base});
        for (const auto &obj : self.vec) {
            Py_VISIT(obj.ptr());
        }
        return 0;
    }

    static int tp_clear(PyObject *self_base) {
        if (should_check_holder_initialization && !py::detail::is_holder_constructed(self_base)) {
            // The holder has not been constructed yet. Skip the traversal to avoid
            // segmentation faults.
            return 0;
        }

        // The actual logic of the tp_clear function goes here.
        auto &self = py::cast<VecOwnsObjs &>(py::handle{self_base});
        for (auto &obj : self.vec) {
            Py_CLEAR(obj.ptr());
        }
        self.vec.clear();
        return 0;
    }

    py::object get_state() const {
        py::list state{};
        for (const auto &item : vec) {
            state.append(item);
        }
        return py::tuple(state);
    }

    static bool get_should_check_holder_initialization() {
        return should_check_holder_initialization;
    }

    static void set_should_check_holder_initialization(bool value) {
        should_check_holder_initialization = value;
    }

    static bool get_should_raise_error_on_set_state() { return should_raise_error_on_set_state; }

    static void set_should_raise_error_on_set_state(bool value) {
        should_raise_error_on_set_state = value;
    }

    static bool should_check_holder_initialization;
    static bool should_raise_error_on_set_state;

private:
    std::vector<py::object> vec{};
};

bool VecOwnsObjs::should_check_holder_initialization = false;
bool VecOwnsObjs::should_raise_error_on_set_state = false;

TEST_SUBMODULE(invalid_holder_access, m) {
    m.doc() = "Test invalid holder access";

#if defined(PYBIND11_CPP14)
    m.def("create_vector", [](const py::iterable &iterable) -> std::unique_ptr<VecOwnsObjs> {
        auto vec = std::make_unique<VecOwnsObjs>();
        for (const auto &item : iterable) {
            vec->append(py::reinterpret_borrow<py::object>(item));
        }
        return vec;
    });
#endif

    py::class_<VecOwnsObjs>(
        m, "VecOwnsObjs", py::custom_type_setup([](PyHeapTypeObject *heap_type) -> void {
            auto *const type = &heap_type->ht_type;
            type->tp_flags |= Py_TPFLAGS_HAVE_GC;
            type->tp_traverse = &VecOwnsObjs::tp_traverse;
            type->tp_clear = &VecOwnsObjs::tp_clear;
        }))
        .def_static("set_should_check_holder_initialization",
                    &VecOwnsObjs::set_should_check_holder_initialization,
                    py::arg("value"))
        .def_static("set_should_raise_error_on_set_state",
                    &VecOwnsObjs::set_should_raise_error_on_set_state,
                    py::arg("value"))
#if defined(PYBIND11_CPP14)
        .def(py::pickle([](const VecOwnsObjs &self) -> py::object { return self.get_state(); },
                        [](const py::object &state) -> std::unique_ptr<VecOwnsObjs> {
                            if (!py::isinstance<py::tuple>(state)) {
                                throw std::runtime_error("Invalid state");
                            }
                            auto vec = std::make_unique<VecOwnsObjs>();
                            if (VecOwnsObjs::get_should_raise_error_on_set_state()) {
                                throw std::runtime_error("raise error on set_state for testing");
                            }
                            for (const auto &item : state) {
                                vec->append(py::reinterpret_borrow<py::object>(item));
                            }
                            return vec;
                        }),
             py::arg("state"))
#endif
        .def("append", &VecOwnsObjs::append, py::arg("obj"))
        .def("is_empty", &VecOwnsObjs::is_empty)
        .def("__setitem__", &VecOwnsObjs::set_item, py::arg("i"), py::arg("obj"))
        .def("__getitem__", &VecOwnsObjs::get_item, py::arg("i"))
        .def("__len__", &VecOwnsObjs::size);
}
