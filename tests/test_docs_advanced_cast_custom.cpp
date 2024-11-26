// #########################################################################
// PLEASE UPDATE docs/advanced/cast/custom.rst IF ANY CHANGES ARE MADE HERE.
// #########################################################################

#include "pybind11_tests.h"

namespace user_space {

struct Point2D {
    double x;
    double y;
};

Point2D negate(const Point2D &point) { return Point2D{-point.x, -point.y}; }

} // namespace user_space

namespace pybind11 {
namespace detail {

template <>
struct type_caster<user_space::Point2D> {
    // This macro inserts a lot of boilerplate code and sets the default type hint to `tuple`
    PYBIND11_TYPE_CASTER(user_space::Point2D, const_name("tuple"));
    // `arg_name` and `return_name` may optionally be used to specify type hints separately for
    // arguments and return values.
    // The signature of our identity function would then look like:
    // `identity(Sequence[float]) -> tuple[float, float]`
    static constexpr auto arg_name = const_name("Sequence[float]");
    static constexpr auto return_name = const_name("tuple[float, float]");

    // C++ -> Python: convert `Point2D` to `tuple[float, float]`. The second and third arguments
    // are used to indicate the return value policy and parent object (for
    // return_value_policy::reference_internal) and are often ignored by custom casters.
    static handle cast(const user_space::Point2D &number, return_value_policy, handle) {
        // Convert x and y components to python float
        auto *x = PyFloat_FromDouble(number.x);
        auto *y = PyFloat_FromDouble(number.y);
        // Check if conversion was successful otherwise clean up references and return null
        if (!x || !y) {
            Py_XDECREF(x);
            Py_XDECREF(y);
            return nullptr;
        }
        // Create tuple from x and y
        auto t = PyTuple_Pack(2, x, y);
        // Decrement references (the tuple now owns x an y)
        Py_DECREF(x);
        Py_DECREF(y);
        return t;
    }

    // Python -> C++: convert a `PyObject` into a `Point2D` and return false upon failure. The
    // second argument indicates whether implicit conversions should be allowed.
    bool load(handle src, bool) {
        // Check if handle is valid Sequence of length 2
        if (!src || PySequence_Check(src.ptr()) == 0 || PySequence_Length(src.ptr()) != 2) {
            return false;
        }
        auto *x = PySequence_GetItem(src.ptr(), 0);
        auto *y = PySequence_GetItem(src.ptr(), 1);
        // Check if values are float or int (both are allowed with float as type hint)
        if (!x || !(PyFloat_Check(x) || PyLong_Check(x)) || !y
            || !(PyFloat_Check(y) || PyLong_Check(y))) {
            Py_XDECREF(x);
            Py_XDECREF(y);
            return false;
        }
        // value is a default constructed Point2D
        value.x = PyFloat_AsDouble(x);
        value.y = PyFloat_AsDouble(y);
        Py_DECREF(x);
        Py_DECREF(y);
        if ((value.x == -1.0 || value.y == -1.0) && PyErr_Occurred()) {
            PyErr_Clear();
            return false;
        }
        return true;
    }
};

} // namespace detail
} // namespace pybind11

// Bind the negate function
TEST_SUBMODULE(docs_advanced_cast_custom, m) { m.def("negate", user_space::negate); }
