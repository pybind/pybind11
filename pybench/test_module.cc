#include <pybind11/pybind11.h>

int add(int i, int j) { return i + j; }

PYBIND11_MODULE(custom, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    std::cout << "Working with " << (void *) (&add) << std::endl;

    m.def("test_me", &add, "A function that adds two numbers");
}
