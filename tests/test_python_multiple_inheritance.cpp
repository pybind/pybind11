#include "pybind11_tests.h"

namespace test_python_multiple_inheritance {

// Copied from:
// https://github.com/google/clif/blob/5718e4d0807fd3b6a8187dde140069120b81ecef/clif/testing/python_multiple_inheritance.h

struct CppBase {
    explicit CppBase(int value) : base_value(value) {}
    int get_base_value() const { return base_value; }
    void reset_base_value(int new_value) { base_value = new_value; }

private:
    int base_value;
};

struct CppDrvd : CppBase {
    explicit CppDrvd(int value) : CppBase(value), drvd_value(value * 3) {}
    int get_drvd_value() const { return drvd_value; }
    void reset_drvd_value(int new_value) { drvd_value = new_value; }

    int get_base_value_from_drvd() const { return get_base_value(); }
    void reset_base_value_from_drvd(int new_value) { reset_base_value(new_value); }

private:
    int drvd_value;
};

struct CppDrv2 : CppDrvd {
    explicit CppDrv2(int value) : CppDrvd(value), drv2_value(value * 5) {}
    int get_drv2_value() const { return drv2_value; }

private:
    int drv2_value;
};

} // namespace test_python_multiple_inheritance

TEST_SUBMODULE(python_multiple_inheritance, m) {
    using namespace test_python_multiple_inheritance;

    py::class_<CppBase>(m, "CppBase")
        .def(py::init<int>())
        .def("get_base_value", &CppBase::get_base_value)
        .def("reset_base_value", &CppBase::reset_base_value);

    py::class_<CppDrvd, CppBase>(m, "CppDrvd")
        .def(py::init<int>())
        .def("get_drvd_value", &CppDrvd::get_drvd_value)
        .def("reset_drvd_value", &CppDrvd::reset_drvd_value)
        .def("get_base_value_from_drvd", &CppDrvd::get_base_value_from_drvd)
        .def("reset_base_value_from_drvd", &CppDrvd::reset_base_value_from_drvd);

#if 1 // This works.
    py::class_<CppDrv2, CppDrvd>(m, "CppDrv2")
#else // Apparent undefined behavior.
    py::class_<CppDrv2, CppDrvd, CppBase>(m, "CppDrv2")
#endif
        .def(py::init<int>())
        .def("get_drv2_value", &CppDrv2::get_drv2_value);
}
