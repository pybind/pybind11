#include "pybind11_tests.h"

#include <iostream>
#include <memory>

template <class T>
using DefaultAllocator = std::allocator<T>;

template <template <class> class Allocator = DefaultAllocator>
struct Base {};

template <template <class> class Allocator = DefaultAllocator>
struct S : public Base<Allocator> {};

// this returns S<DefaultAllocator> even though we register
// the type S<std::allocator>
// MSVC and Clang on Windows failed here to detect the registered type
S<> make_S() {
    std::cout << "in make_S()" << std::endl;
    return S<>{};
}

TEST_SUBMODULE(template_alias_base, m) {
    py::class_<Base<std::allocator>>(m, "B_std").def(py::init());
    py::class_<S<std::allocator>, Base<std::allocator>>(m, "S_std").def(py::init());

    m.def("make_S", &make_S);
}
