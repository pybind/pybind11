#include <pybind11/functional.h>

#include "pybind11_tests.h"

namespace pybind11 {
namespace detail {

template <>
class type_caster<PyObject> {
public:
    static constexpr auto name = const_name("PyObject *");

    static handle cast(PyObject *src, return_value_policy policy, handle parent) {
        if (src == nullptr) {
            // NEEDS TEST
            throw error_already_set();
        }
        if (PyErr_Occurred()) {
            // NEEDS TEST
            raise_from(PyExc_SystemError, "src != nullptr but PyErr_Occurred()");
        }
        if (policy == return_value_policy::take_ownership) {
            return src;
        }
        if (policy == return_value_policy::reference_internal) {
            // NEEDS TEST
            keep_alive_impl(src, parent);
        }
        Py_INCREF(src);
        return src;
    }

    bool load(handle src, bool) {
        value = reinterpret_borrow<object>(src);
        return true;
    }

    template <typename T>
    using cast_op_type = PyObject *;

    explicit operator PyObject *() { return value.ptr(); }

private:
    object value;
};

} // namespace detail
} // namespace pybind11

TEST_SUBMODULE(type_caster_pyobject_ptr, m) {
    m.def("cast_from_PyObject_ptr", []() {
        PyObject *ptr = PyLong_FromLongLong(6758L);
        py::object retval = py::cast(ptr, py::return_value_policy::take_ownership);
        return retval;
    });
    m.def("cast_to_PyObject_ptr", [](py::handle obj) {
        auto *ptr = py::cast<PyObject *>(obj);
        return bool(PyTuple_CheckExact(ptr));
    });

    m.def(
        "return_PyObject_ptr",
        []() { return PyLong_FromLongLong(2314L); },
        py::return_value_policy::take_ownership);
    m.def("pass_PyObject_ptr", [](PyObject *obj) { return bool(PyTuple_CheckExact(obj)); });

    m.def("call_callback_with_object_return",
          [](const std::function<py::object(int mode)> &cb, int mode) { return cb(mode); });
    m.def("call_callback_with_handle_return",
          [](const std::function<py::handle(int mode)> &cb, int mode) { return cb(mode); });
    //
    m.def("call_callback_with_PyObject_ptr_return",
          [](const std::function<PyObject *(int mode)> &cb, int mode) { return cb(mode); });
    m.def("call_callback_with_PyObject_ptr_arg",
          [](const std::function<bool(PyObject *)> &cb, py::handle obj) { return cb(obj.ptr()); });
}
