#include <pybind11/functional.h>

#include "pybind11_tests.h"

namespace pybind11 {
namespace detail {

template <typename T, typename U>
using is_same_ignoring_cvref = std::is_same<detail::remove_cvref_t<T>, U>;

template <>
class type_caster<PyObject> {
public:
    static constexpr auto name = const_name("PyObject *");

    // This overload is purely to guard against accidents.
    template <typename T,
              detail::enable_if_t<!is_same_ignoring_cvref<T, PyObject *>::value, int> = 0>
    static handle cast(T &&, return_value_policy, handle /*parent*/) {
        static_assert(is_same_ignoring_cvref<T, PyObject *>::value,
                      "Invalid C++ type T for to-Python conversion (type_caster<PyObject>).");
        return nullptr; // Unreachable.
    }

    static handle cast(PyObject *src, return_value_policy policy, handle /*parent*/) {
        if (src == nullptr) {
            throw error_already_set();
        }
        if (PyErr_Occurred()) {
            raise_from(PyExc_SystemError, "src != nullptr but PyErr_Occurred()");
            throw error_already_set();
        }
        if (policy == return_value_policy::take_ownership) {
            return src;
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
