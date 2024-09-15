// Copyright (c) 2024 The pybind Community.

// THIS MUST STAY AT THE TOP!
#include <pybind11/compat/pybind11_conduit_v1.h> // VERY light-weight dependency.

#include "test_cpp_conduit_traveler_types.h"

#include <Python.h>

namespace {

extern "C" PyObject *wrapGetLuggage(PyObject * /*self*/, PyObject *traveler) {
    const auto *cpp_traveler = pybind11_conduit_v1::get_type_pointer_ephemeral<
        pybind11_tests::test_cpp_conduit::Traveler>(traveler);
    if (cpp_traveler == nullptr) {
        return nullptr;
    }
    return PyUnicode_FromString(cpp_traveler->luggage.c_str());
}

extern "C" PyObject *wrapGetPoints(PyObject * /*self*/, PyObject *premium_traveler) {
    const auto *cpp_premium_traveler = pybind11_conduit_v1::get_type_pointer_ephemeral<
        pybind11_tests::test_cpp_conduit::PremiumTraveler>(premium_traveler);
    if (cpp_premium_traveler == nullptr) {
        return nullptr;
    }
    return PyLong_FromLong(static_cast<long>(cpp_premium_traveler->points));
}

PyMethodDef ThisMethodDef[] = {{"GetLuggage", wrapGetLuggage, METH_O, nullptr},
                               {"GetPoints", wrapGetPoints, METH_O, nullptr},
                               {nullptr, nullptr, 0, nullptr}};

struct PyModuleDef ThisModuleDef = {
    PyModuleDef_HEAD_INIT, // m_base
    "exo_planet_c_api",    // m_name
    nullptr,               // m_doc
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
