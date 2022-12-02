#include <pybind11/pybind11.h>

#define USE_SH

#if defined(USE_SH)
#    include <pybind11/smart_holder.h>
#endif

#include "pybind11_tests.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace test_mi_debug {

struct Base0 {
    virtual ~Base0() = default;
};

struct Base1 {
    virtual ~Base1() = default;
    std::vector<int> vec = {1, 2, 3, 4, 5};
};

struct Derived : Base1, Base0 {
    ~Derived() override = default;
};

} // namespace test_mi_debug

#if defined(USE_SH)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(test_mi_debug::Base0)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(test_mi_debug::Base1)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(test_mi_debug::Derived)
#endif

TEST_SUBMODULE(mi_debug, m) {
    using namespace test_mi_debug;

#if defined(USE_SH)
    py::classh<Base0> bs0(m, "Base0");
    py::classh<Base1> bs1(m, "Base1");
    py::classh<Derived, Base1, Base0>(m, "Derived").def(py::init<>());
#else
    py::class_<Base0, std::shared_ptr<Base0>> bs0(m, "Base0");
    py::class_<Base1, std::shared_ptr<Base1>> bs1(m, "Base1");
    py::class_<Derived, std::shared_ptr<Derived>, Base1, Base0>(m, "Derived").def(py::init<>());
#endif

    m.def("make_derived_as_base0", []() -> std::shared_ptr<Base0> {
        auto ret_der = std::make_shared<Derived>();
        auto ret = std::dynamic_pointer_cast<Base0>(ret_der);
        return ret;
    });

    // class_ OK
    // classh FAIL
    m.def("get_vec_size_raw_ptr_base0", [](const Base0 *obj) -> std::size_t {
        auto obj_der = dynamic_cast<const Derived *>(obj);
        if (obj_der == nullptr) {
            return 0;
        }
        return obj_der->vec.size();
    });

    // class_ OK
    // classh FAIL
    m.def("get_vec_size_raw_ptr_derived", [](const Derived *obj) { return obj->vec.size(); });

    // class_ OK
    // classh FAIL
    m.def("get_vec_size_shared_ptr_derived",
          [](const std::shared_ptr<Derived> &obj) { return obj->vec.size(); });
}
