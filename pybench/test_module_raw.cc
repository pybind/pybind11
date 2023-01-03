#include <Python.h>

PyObject *func(PyObject *self, PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames) {
    return PyLong_FromLong(PyLong_AsLong(args[0]) + PyLong_AsLong(args[1]));
}

PyMethodDef methods[2] = {};
PyModuleDef df = {};

PyMODINIT_FUNC PyInit_raw_custom(void) {
    df.m_base = PyModuleDef_HEAD_INIT;
    df.m_name = "raw_test";
    df.m_doc = "what";
    df.m_size = 0;
    df.m_methods = methods;

    methods[0].ml_name = "test_me";
    methods[0].ml_doc = "doc of test me";
    methods[0].ml_flags = METH_FASTCALL | METH_KEYWORDS;
    methods[0].ml_meth = (PyCFunction) func;

    PyObject *m = PyModule_Create(&df);
    if (m == NULL)
        return NULL;

    return m;
}
