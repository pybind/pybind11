#include <Python.h> // THIS MUST STAY AT THE TOP!

// DO NOT USE ANY OTHER pybind11 HEADERS HERE!
#include <pybind11/detail/platform_abi_id.h>

#include "test_cpp_transporter_traveler_types.h"

namespace {

extern "C" PyObject *wrapGetLuggage(PyObject *, PyObject *) {
    return PyUnicode_FromString("TODO");
}

PyDoc_STRVAR(ThisModuleDoc, "Uses only the plain CPython API.");

PyMethodDef ThisMethodDef[]
    = {{"GetLuggage", wrapGetLuggage, METH_VARARGS, nullptr}, {nullptr, nullptr, 0, nullptr}};

struct PyModuleDef ThisModuleDef = {
    PyModuleDef_HEAD_INIT, // m_base
    "exo_planet_c_api",    // m_name
    ThisModuleDoc,         // m_doc
    -1,                    // m_size
    ThisMethodDef,         // m_methods
    nullptr,               // m_slots
    nullptr,               // m_traverse
    nullptr,               // m_clear
    nullptr                // m_free
};

} // namespace

#if defined(WIN32) || defined(_WIN32)
#    define EXO_PLANET_C_API_EXPORT __declspec(dllexport)
#else
#    define EXO_PLANET_C_API_EXPORT __attribute__((visibility("default")))
#endif

extern "C" EXO_PLANET_C_API_EXPORT PyObject *PyInit_exo_planet_c_api() {
    PyObject *m = PyModule_Create(&ThisModuleDef);
    if (m == nullptr) {
        return nullptr;
    }
    return m;
}
