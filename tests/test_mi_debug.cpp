#include <pybind11/pybind11.h>

#include <iostream>
#include <memory>
#include <vector>

// The first base class.
struct Base0 {
    virtual ~Base0() = default;
};

using Base0Ptr = std::shared_ptr<Base0>;

// The second base class.
struct Base1 {
    virtual ~Base1() = default;
    std::vector<int> vec = {1, 2, 3, 4, 5};
};

using Base1Ptr = std::shared_ptr<Base1>;

// The derived class.
struct Derived : Base1, Base0 {
    ~Derived() override = default;
};

using DerivedPtr = std::shared_ptr<Derived>;

PYBIND11_MODULE(example, m) {
    // Expose the bases.
    pybind11::class_<Base0, Base0Ptr> bs0(m, "Base0");
    pybind11::class_<Base1, Base1Ptr> bs1(m, "Base1");
    // Expose the derived class.
    pybind11::class_<Derived, DerivedPtr, Base0, Base1>(m, "Derived").def(pybind11::init<>());

    // A helper that returns a pointer to base.
    m.def("make_object", []() -> Base0Ptr {
        auto ret_der = std::make_shared<Derived>();
        std::cout << "ret der ptr: " << ret_der.get() << std::endl;
        auto ret = Base0Ptr(ret_der);
        std::cout << "ret base ptr: " << ret.get() << std::endl;
        return ret;
    });

    // A helper that accepts in input a pointer to derived.
    m.def("get_object_vec_size", [](const DerivedPtr &object) {
        std::cout << "der ptr: " << object.get() << std::endl;
        std::cout << object->vec.size() << std::endl;
        return object->vec.size();
    });
}
