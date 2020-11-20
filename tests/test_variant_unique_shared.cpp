#include <memory>
#include <variant>

#include "pybind11_tests.h"

namespace pybind11_tests {

// Could this be a holder for a `class_`-like `vclass`?
// To enable passing of unique_ptr as in pure C++.
template <typename T> class vptr_holder {
  public:
    explicit vptr_holder(T *ptr = nullptr) : vptr_{std::unique_ptr<T>(ptr)} {}
    explicit vptr_holder(std::unique_ptr<T> u) : vptr_{std::move(u)} {}
    explicit vptr_holder(std::shared_ptr<T> s) : vptr_{s} {}

    int ownership_type() const {
        if (std::get_if<0>(&vptr_)) {
            return 0;
        }
        if (std::get_if<1>(&vptr_)) {
            return 1;
        }
        return -1;
    }

    T *get() {
        auto u = std::get_if<0>(&vptr_);
        if (u) {
            return u->get();
        }
        auto s = std::get_if<1>(&vptr_);
        if (s) {
            return s->get();
        }
        return nullptr;
    }

    std::unique_ptr<T> get_unique() {
        auto u = std::get_if<0>(&vptr_);
        if (u) {
            return std::move(*u);
        }
        throw std::runtime_error("get_unique failure.");
    }

    std::shared_ptr<T> get_shared() {
        auto s = std::get_if<1>(&vptr_);
        if (s) {
            return *s;
        }
        auto u = std::get_if<0>(&vptr_);
        if (u) {
            auto result = std::shared_ptr<T>(std::move(*u));
            vptr_ = result;
            return result;
        }
        throw std::runtime_error("get_shared failure.");
    }

  private:
    std::variant<std::unique_ptr<T>, std::shared_ptr<T>> vptr_;
};

vptr_holder<double> from_raw() { return vptr_holder<double>{new double{3}}; }

vptr_holder<double> from_unique() {
    return vptr_holder<double>{std::unique_ptr<double>(new double{5})};
}

vptr_holder<double> from_shared() {
    return vptr_holder<double>{std::shared_ptr<double>(new double{7})};
}

TEST_SUBMODULE(variant_unique_shared, m) {

    m.def("from_raw", from_raw);
    m.def("from_unique", from_unique);
    m.def("from_shared", from_shared);

    py::class_<vptr_holder<double>>(m, "vptr_holder_double")
        .def(py::init<>())
        .def("ownership_type", &vptr_holder<double>::ownership_type)
        .def("get_value",
             [](vptr_holder<double> &v) {
                 auto p = v.get();
                 if (p)
                     return *p;
                 return -1.;
             })
        .def("get_unique",
             [](vptr_holder<double> &v) {
                 v.get_unique();
                 return;
             })
        .def("get_shared", [](vptr_holder<double> &v) {
            v.get_shared();
            return;
        });
}

} // namespace pybind11_tests
