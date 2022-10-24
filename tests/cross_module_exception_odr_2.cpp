#include "pybind11/pybind11.h"

#include <exception>

namespace cross_module_exception_odr {

class PYBIND11_EXPORT_EXCEPTION evolving : public std::exception {
public:
    const char *what() const noexcept override { return "v2"; }
};

void raise_evolving_from_module_2() { throw evolving(); }

} // namespace cross_module_exception_odr

PYBIND11_MODULE(cross_module_exception_odr_2, m) {
    using namespace cross_module_exception_odr;
    namespace py = pybind11;

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const evolving &e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    m.def("raise_evolving", []() { throw evolving(); });
    m.def("get_raise_evolving_from_module_2_capsule", []() {
        return py::capsule(reinterpret_cast<void *>(&raise_evolving_from_module_2),
                           "raise_evolving_from_module_2_function_pointer");
    });
}
