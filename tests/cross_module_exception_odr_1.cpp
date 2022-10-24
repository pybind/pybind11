#include "pybind11/pybind11.h"

#include <exception>

namespace cross_module_exception_odr {

class PYBIND11_EXPORT_EXCEPTION evolving : public std::runtime_error {
public:
    explicit evolving(const std::string &msg) : std::runtime_error("v1:" + msg) {}
};

} // namespace cross_module_exception_odr

PYBIND11_MODULE(cross_module_exception_odr_1, m) {
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

    m.def("raise_evolving", [](const std::string &msg) { throw evolving(msg); });
    m.def("raise_evolving_from_module_2", [](const py::capsule &cap) {
        auto f = reinterpret_cast<void (*)()>(cap.get_pointer<void>());
        f();
    });
}
