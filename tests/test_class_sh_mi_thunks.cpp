#include <pybind11/pybind11.h>
#include <pybind11/smart_holder.h>

#include "pybind11_tests.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace test_class_sh_mi_thunks {

struct Base0 {
    virtual ~Base0() = default;
    Base0() = default;
    Base0(const Base0 &) = delete;
};

struct Base1 {
    virtual ~Base1() = default;
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

PYBIND11_SMART_HOLDER_TYPE_CASTERS(test_class_sh_mi_thunks::Base0)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(test_class_sh_mi_thunks::Base1)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(test_class_sh_mi_thunks::Derived)

TEST_SUBMODULE(class_sh_mi_thunks, m) {
    using namespace test_class_sh_mi_thunks;

    m.def("ptrdiff_derived_base0", []() {
        auto drvd = std::unique_ptr<Derived>(new Derived{});
        auto base0 = dynamic_cast<Base0 *>(drvd.get());
        return std::ptrdiff_t(reinterpret_cast<char *>(drvd.get())
                              - reinterpret_cast<char *>(base0));
    });

    py::classh<Base0> bs0(m, "Base0");
    py::classh<Base1> bs1(m, "Base1");
    py::classh<Derived, Base1, Base0>(m, "Derived");

    m.def(
        "get_derived_as_base0_raw_ptr",
        []() {
            auto *ret_der = new Derived{};
            auto *ret = dynamic_cast<Base0 *>(ret_der);
            return ret;
        },
        py::return_value_policy::take_ownership);

    m.def("get_derived_as_base0_shared_ptr", []() -> std::shared_ptr<Base0> {
        auto ret_der = std::make_shared<Derived>();
        auto ret = std::dynamic_pointer_cast<Base0>(ret_der);
        return ret;
    });

    m.def("get_derived_as_base0_unique_ptr", []() -> std::unique_ptr<Base0> {
        auto ret_der = std::unique_ptr<Derived>(new Derived{});
        auto ret = std::unique_ptr<Base0>(std::move(ret_der));
        return ret;
    });

    m.def("vec_size_base0_raw_ptr", [](const Base0 *obj) -> std::size_t {
        const auto *obj_der = dynamic_cast<const Derived *>(obj);
        if (obj_der == nullptr) {
            return 0;
        }
        return obj_der->vec.size();
    });

    m.def("vec_size_base0_shared_ptr", [](const std::shared_ptr<Base0> &obj) -> std::size_t {
        const auto obj_der = std::dynamic_pointer_cast<Derived>(obj);
        if (!obj_der) {
            return 0;
        }
        return obj_der->vec.size();
    });

    m.def("vec_size_base0_unique_ptr", [](std::unique_ptr<Base0> obj) -> std::size_t {
        const auto *obj_der = dynamic_cast<const Derived *>(obj.get());
        if (obj_der == nullptr) {
            return 0;
        }
        return obj_der->vec.size();
    });
}
