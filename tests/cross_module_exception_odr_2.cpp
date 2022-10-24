#include "pybind11/pybind11.h"

#include <exception>

namespace cross_module_exception_odr {

#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4275)
#endif

class PYBIND11_EXPORT_EXCEPTION evolving : public std::exception {
public:
    const char *what() const noexcept override { return "v2"; }
};

#if defined(_MSC_VER)
#    pragma warning(pop)
#endif

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
