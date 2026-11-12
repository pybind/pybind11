#include <pybind11/functional.h>

namespace py = pybind11;

int add_one(int value) { return value + 1; }

PYBIND11_MODULE(TEST_FUNCTION_RECORD_MODULE, m, py::mod_gil_not_used()) {
#if defined(PYBIND11_OPAQUE_PYOBJECT)
    m.attr("opaque_function_record") = true;
#else
    m.attr("opaque_function_record") = false;
#endif
    m.def("add_one", &add_one);
    m.def("call_callback", [](const std::function<int(int)> &callback) {
        return py::make_tuple(callback(41), callback.target<int (*)(int)>() != nullptr);
    });
}
