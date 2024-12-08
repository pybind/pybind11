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
    // The signature of our negate function would then look like:
    // `negate(Sequence[float]) -> tuple[float, float]`
    static constexpr auto arg_name = const_name("Sequence[float]");
    static constexpr auto return_name = const_name("tuple[float, float]");

    // C++ -> Python: convert `Point2D` to `tuple[float, float]`. The second and third arguments
    // are used to indicate the return value policy and parent object (for
    // return_value_policy::reference_internal) and are often ignored by custom casters.
    // The return value should reflect the type hint specified by `return_name`.
    static handle
    cast(const user_space::Point2D &number, return_value_policy /*policy*/, handle /*parent*/) {
        return py::make_tuple(number.x, number.y).release();
    }

    // Python -> C++: convert a `PyObject` into a `Point2D` and return false upon failure. The
    // second argument indicates whether implicit conversions should be allowed.
    // The accepted types should reflect the type hint specified by `arg_name`.
    bool load(handle src, bool /*convert*/) {
        // Check if handle is a Sequence
        if (!py::isinstance<py::sequence>(src)) {
            return false;
        }
        auto seq = py::reinterpret_borrow<py::sequence>(src);
        // Check if exactly two values are in the Sequence
        if (seq.size() != 2) {
            return false;
        }
        // Check if each element is either a float or an int
        for (auto item : seq) {
            if (!py::isinstance<py::float_>(item) && !py::isinstance<py::int_>(item)) {
                return false;
            }
        }
        value.x = seq[0].cast<double>();
        value.y = seq[1].cast<double>();
        return true;
    }
};

} // namespace detail
} // namespace pybind11

// Bind the negate function
TEST_SUBMODULE(docs_advanced_cast_custom, m) { m.def("negate", user_space::negate); }
