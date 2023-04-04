#include <pybind11/functional.h>
#include <pybind11/type_caster_pyobject_ptr.h>

#include "pybind11_tests.h"

TEST_SUBMODULE(type_caster_pyobject_ptr, m) {
    m.def("cast_from_pyobject_ptr", []() {
        PyObject *ptr = PyLong_FromLongLong(6758L);
        return py::cast(ptr, py::return_value_policy::take_ownership);
    });
    m.def("cast_to_pyobject_ptr", [](py::handle obj) {
        auto rc1 = obj.ref_count();
        auto *ptr = py::cast<PyObject *>(obj);
        auto rc2 = obj.ref_count();
        if (rc2 != rc1 + 1) {
            return -1;
        }
        return 100 - py::reinterpret_steal<py::object>(ptr).attr("value").cast<int>();
    });

    m.def(
        "return_pyobject_ptr",
        []() { return PyLong_FromLongLong(2314L); },
        py::return_value_policy::take_ownership);
    m.def("pass_pyobject_ptr", [](PyObject *ptr) {
        return 200 - py::reinterpret_borrow<py::object>(ptr).attr("value").cast<int>();
    });

    m.def("call_callback_with_object_return",
          [](const std::function<py::object(int)> &cb, int value) { return cb(value); });
    m.def(
        "call_callback_with_pyobject_ptr_return",
        [](const std::function<PyObject *(int)> &cb, int value) { return cb(value); },
        py::return_value_policy::take_ownership);
    m.def(
        "call_callback_with_pyobject_ptr_arg",
        [](const std::function<int(PyObject *)> &cb, py::handle obj) { return cb(obj.ptr()); },
        py::arg("cb"), // This triggers return_value_policy::automatic_reference
        py::arg("obj"));

    m.def("cast_to_pyobject_ptr_nullptr", [](bool set_error) {
        if (set_error) {
            PyErr_SetString(PyExc_RuntimeError, "Reflective of healthy error handling.");
        }
        PyObject *ptr = nullptr;
        py::cast(ptr);
    });

    m.def("cast_to_pyobject_ptr_non_nullptr_with_error_set", []() {
        PyErr_SetString(PyExc_RuntimeError, "Reflective of unhealthy error handling.");
        py::cast(Py_None);
    });

#ifdef PYBIND11_NO_COMPILE_SECTION // Change to ifndef for manual testing.
    {
        PyObject *ptr = nullptr;
        (void) py::cast(*ptr);
    }
#endif
}
