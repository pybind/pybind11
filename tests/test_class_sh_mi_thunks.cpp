#include "pybind11_tests.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace test_class_sh_mi_thunks {

// For general background: https://shaharmike.com/cpp/vtable-part2/
// C++ vtables - Part 2 - Multiple Inheritance
// ... the compiler creates a 'thunk' method that corrects `this` ...

// This test was added under PR #4380

struct Base0 {
    virtual ~Base0() = default;
    Base0() = default;
    Base0(const Base0 &) = delete;
};

struct Base1 {
    virtual ~Base1() = default;
    // Using `vector` here because it is known to make this test very sensitive to bugs.
    std::vector<int> vec = {1, 2, 3, 4, 5};
    Base1() = default;
    Base1(const Base1 &) = delete;
};

struct Derived : Base1, Base0 {
    ~Derived() override = default;
    Derived() = default;
    Derived(const Derived &) = delete;
};

// ChatGPT-generated Diamond added under PR #5836

struct VBase {
    VBase() = default;
    VBase(const VBase &) = default; // silence -Wdeprecated-copy-with-dtor
    VBase &operator=(const VBase &) = default;
    VBase(VBase &&) = default;
    VBase &operator=(VBase &&) = default;
    virtual ~VBase() = default;
    virtual int ping() const { return 1; }
    int vbase_tag = 42; // ensure it's not empty
};

// Make the virtual bases non-empty and (likely) differently sized.
// The test does *not* require different sizes; we only want to avoid "all at offset 0".
// If a compiler/ABI still places the virtual base at offset 0, our test logs that via
// test_virtual_base_at_offset_0() and continues.
struct Left : virtual VBase {
    char pad_l[4]; // small, typically 4 + padding
    ~Left() override = default;
};
struct Right : virtual VBase {
    char pad_r[24]; // larger, to differ from Left
    ~Right() override = default;
};

struct Diamond : Left, Right {
    Diamond() = default;
    Diamond(const Diamond &) = default;
    ~Diamond() override = default;
    int ping() const override { return 7; }
    int self_tag = 99;
};

VBase *make_diamond_as_vbase_raw_ptr() {
    auto *ptr = new Diamond;
    return ptr; // upcast
}

std::shared_ptr<VBase> make_diamond_as_vbase_shared_ptr() {
    auto shptr = std::make_shared<Diamond>();
    return shptr; // upcast
}

std::unique_ptr<VBase> make_diamond_as_vbase_unique_ptr() {
    auto uqptr = std::unique_ptr<Diamond>(new Diamond);
    return uqptr; // upcast
}

// For diagnostics
struct DiamondAddrs {
    uintptr_t as_self;
    uintptr_t as_vbase;
    uintptr_t as_left;
    uintptr_t as_right;
};

DiamondAddrs diamond_addrs() {
    auto sp = std::make_shared<Diamond>();
    return DiamondAddrs{reinterpret_cast<uintptr_t>(sp.get()),
                        reinterpret_cast<uintptr_t>(static_cast<VBase *>(sp.get())),
                        reinterpret_cast<uintptr_t>(static_cast<Left *>(sp.get())),
                        reinterpret_cast<uintptr_t>(static_cast<Right *>(sp.get()))};
}

// Animal-Cat-Tiger reproducer copied from PR #5796
// clone_raw_ptr, clone_unique_ptr added under PR #5836

class Animal {
public:
    Animal() = default;
    Animal(const Animal &) = default;
    Animal &operator=(const Animal &) = default;
    virtual Animal *clone_raw_ptr() const = 0;
    virtual std::shared_ptr<Animal> clone_shared_ptr() const = 0;
    virtual std::unique_ptr<Animal> clone_unique_ptr() const = 0;
    virtual ~Animal() = default;
};

class Cat : virtual public Animal {
public:
    Cat() = default;
    Cat(const Cat &) = default;
    Cat &operator=(const Cat &) = default;
    ~Cat() override = default;
};

class Tiger : virtual public Cat {
public:
    Tiger() = default;
    Tiger(const Tiger &) = default;
    Tiger &operator=(const Tiger &) = default;
    ~Tiger() override = default;
    Animal *clone_raw_ptr() const override {
        return new Tiger(*this); // upcast
    }
    std::shared_ptr<Animal> clone_shared_ptr() const override {
        return std::make_shared<Tiger>(*this); // upcast
    }
    std::unique_ptr<Animal> clone_unique_ptr() const override {
        return std::unique_ptr<Tiger>(new Tiger(*this)); // upcast
    }
};

} // namespace test_class_sh_mi_thunks

TEST_SUBMODULE(class_sh_mi_thunks, m) {
    using namespace test_class_sh_mi_thunks;

    m.def("ptrdiff_drvd_base0", []() {
        auto drvd = std::unique_ptr<Derived>(new Derived);
        auto *base0 = dynamic_cast<Base0 *>(drvd.get());
        return std::ptrdiff_t(reinterpret_cast<char *>(drvd.get())
                              - reinterpret_cast<char *>(base0));
    });

    py::classh<Base0>(m, "Base0");
    py::classh<Base1>(m, "Base1");
    py::classh<Derived, Base1, Base0>(m, "Derived");

    m.def(
        "get_drvd_as_base0_raw_ptr",
        []() {
            auto *drvd = new Derived;
            auto *base0 = dynamic_cast<Base0 *>(drvd);
            return base0;
        },
        py::return_value_policy::take_ownership);

    m.def("get_drvd_as_base0_shared_ptr", []() {
        auto drvd = std::make_shared<Derived>();
        auto base0 = std::dynamic_pointer_cast<Base0>(drvd);
        return base0;
    });

    m.def("get_drvd_as_base0_unique_ptr", []() {
        auto drvd = std::unique_ptr<Derived>(new Derived);
        auto base0 = std::unique_ptr<Base0>(std::move(drvd));
        return base0;
    });

    m.def("vec_size_base0_raw_ptr", [](const Base0 *obj) {
        const auto *obj_der = dynamic_cast<const Derived *>(obj);
        if (obj_der == nullptr) {
            return std::size_t(0);
        }
        return obj_der->vec.size();
    });

    m.def("vec_size_base0_shared_ptr", [](const std::shared_ptr<Base0> &obj) -> std::size_t {
        const auto obj_der = std::dynamic_pointer_cast<Derived>(obj);
        if (!obj_der) {
            return std::size_t(0);
        }
        return obj_der->vec.size();
    });

    m.def("vec_size_base0_unique_ptr", [](std::unique_ptr<Base0> obj) -> std::size_t {
        const auto *obj_der = dynamic_cast<const Derived *>(obj.get());
        if (obj_der == nullptr) {
            return std::size_t(0);
        }
        return obj_der->vec.size();
    });

    py::class_<VBase, py::smart_holder>(m, "VBase").def("ping", &VBase::ping);

    py::class_<Left, VBase, py::smart_holder>(m, "Left");
    py::class_<Right, VBase, py::smart_holder>(m, "Right");

    py::class_<Diamond, Left, Right, py::smart_holder>(m, "Diamond", py::multiple_inheritance())
        .def(py::init<>())
        .def("ping", &Diamond::ping);

    m.def("make_diamond_as_vbase_raw_ptr",
          &make_diamond_as_vbase_raw_ptr,
          py::return_value_policy::take_ownership);
    m.def("make_diamond_as_vbase_shared_ptr", &make_diamond_as_vbase_shared_ptr);
    m.def("make_diamond_as_vbase_unique_ptr", &make_diamond_as_vbase_unique_ptr);

    py::class_<DiamondAddrs, py::smart_holder>(m, "DiamondAddrs")
        .def_readonly("as_self", &DiamondAddrs::as_self)
        .def_readonly("as_vbase", &DiamondAddrs::as_vbase)
        .def_readonly("as_left", &DiamondAddrs::as_left)
        .def_readonly("as_right", &DiamondAddrs::as_right);

    m.def("diamond_addrs", &diamond_addrs);

    py::classh<Animal>(m, "Animal");
    py::classh<Cat, Animal>(m, "Cat");
    py::classh<Tiger, Cat>(m, "Tiger", py::multiple_inheritance())
        .def(py::init<>())
        .def("clone_raw_ptr", &Tiger::clone_raw_ptr)
        .def("clone_shared_ptr", &Tiger::clone_shared_ptr)
        .def("clone_unique_ptr", &Tiger::clone_unique_ptr);
}
