#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
namespace py = pybind11;

PyObject *make_module() {
    py::module m("test_cmake_build");

    m.def("add", [](int i, int j) { return i + j; });

    return m.ptr();
}

int main(int argc, char *argv[]) {
    if (argc != 2)
        throw std::runtime_error("Expected test.py file as the first argument");
    auto test_py_file = argv[1];

    PyImport_AppendInittab("test_cmake_build", &make_module);
    Py_Initialize();
    {
        auto m = py::module::import("test_cmake_build");
        if (m.attr("add")(1, 2).cast<int>() != 3)
            throw std::runtime_error("embed.cpp failed");

        auto globals = py::module::import("__main__").attr("__dict__");
        py::module::import("sys").attr("argv") = py::make_tuple("test.py", "embed.cpp");
        py::eval_file(test_py_file, globals);
    }
    Py_Finalize();
}
