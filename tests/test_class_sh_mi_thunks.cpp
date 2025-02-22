#include "pybind11_tests.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace test_class_sh_mi_thunks {

// For general background: https://shaharmike.com/cpp/vtable-part2/
// C++ vtables - Part 2 - Multiple Inheritance
// ... the compiler creates a 'thunk' method that corrects `this` ...

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
}
