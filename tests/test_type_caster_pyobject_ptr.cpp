#include <pybind11/functional.h>
#include <pybind11/type_caster_pyobject_ptr.h>

#include "pybind11_tests.h"

TEST_SUBMODULE(type_caster_pyobject_ptr, m) {
    m.def("cast_from_pyobject_ptr", []() {
        PyObject *ptr = PyLong_FromLongLong(6758L);
        py::object retval = py::cast(ptr, py::return_value_policy::take_ownership);
        return retval;
    });
    m.def("cast_to_pyobject_ptr", [](py::handle obj) {
        auto *ptr = py::cast<PyObject *>(obj);
        return bool(PyTuple_CheckExact(ptr));
    });

    m.def(
        "return_pyobject_ptr",
        []() { return PyLong_FromLongLong(2314L); },
        py::return_value_policy::take_ownership);
    m.def("pass_pyobject_ptr", [](PyObject *obj) { return bool(PyTuple_CheckExact(obj)); });

    m.def("call_callback_with_object_return",
          [](const std::function<py::object(int mode)> &cb, int mode) { return cb(mode); });
    m.def("call_callback_with_handle_return",
          [](const std::function<py::handle(int mode)> &cb, int mode) { return cb(mode); });
    //
    m.def("call_callback_with_pyobject_ptr_return",
          [](const std::function<PyObject *(int mode)> &cb, int mode) { return cb(mode); });
    m.def("call_callback_with_pyobject_ptr_arg",
          [](const std::function<bool(PyObject *)> &cb, py::handle obj) { return cb(obj.ptr()); });

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
