#include <pybind11/smart_holder.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace hc { // holder comparison

template <int Serial>
struct number_bucket {
    std::vector<double> data;

    explicit number_bucket(std::size_t data_size = 0) : data(data_size, 1.0) {}

    double sum() const {
        std::size_t n   = 0;
        double s        = 0;
        const double *a = &*data.begin();
        const double *e = &*data.end();
        while (a != e) {
            s += *a++;
            n++;
        }
        if (n != data.size()) {
            throw std::runtime_error("Internal consistency failure (sum).");
        }
        return s;
    }

    std::size_t add(const number_bucket &other) {
        if (other.data.size() != data.size()) {
            throw std::runtime_error("Incompatible data sizes.");
        }
        std::size_t n   = 0;
        double *a       = &*data.begin();
        const double *e = &*data.end();
        const double *b = &*other.data.begin();
        while (a != e) {
            *a++ += *b++;
            n++;
        }
        return n;
    }

private:
    number_bucket(const number_bucket &) = delete;
    number_bucket(number_bucket &&)      = delete;
    number_bucket &operator=(const number_bucket &) = delete;
    number_bucket &operator=(number_bucket &&) = delete;
};

namespace py = pybind11;

template <typename WrappedType, typename HolderType>
void wrap_number_bucket(py::module m, const char *class_name) {
    py::class_<WrappedType, HolderType>(m, class_name)
        .def(py::init<std::size_t>(), py::arg("data_size") = 0)
        .def("sum", &WrappedType::sum)
        .def("add", &WrappedType::add, py::arg("other"));
}

template <typename T>
class padded_unique_ptr {
    std::unique_ptr<T> ptr;
    char padding[sizeof(py::smart_holder) - sizeof(std::unique_ptr<T>)];

public:
    padded_unique_ptr(T *p) : ptr(p) {}
    T *get() { return ptr.get(); }
};

using nb_up = number_bucket<0>;
using nb_sp = number_bucket<1>;
using nb_pu = number_bucket<2>;
using nb_sh = number_bucket<3>;

static_assert(sizeof(padded_unique_ptr<nb_pu>) == sizeof(py::smart_holder),
              "Unexpected sizeof mismatch.");

} // namespace hc

PYBIND11_DECLARE_HOLDER_TYPE(T, hc::padded_unique_ptr<T>);

PYBIND11_SMART_POINTER_HOLDER_TYPE_CASTERS(hc::nb_up, std::unique_ptr<hc::nb_up>)
PYBIND11_SMART_POINTER_HOLDER_TYPE_CASTERS(hc::nb_sp, std::shared_ptr<hc::nb_sp>)
PYBIND11_SMART_POINTER_HOLDER_TYPE_CASTERS(hc::nb_pu, hc::padded_unique_ptr<hc::nb_pu>)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(hc::nb_sh)

PYBIND11_MODULE(pybind11_ubench_holder_comparison, m) {
    using namespace hc;
    wrap_number_bucket<nb_up, std::unique_ptr<nb_up>>(m, "number_bucket_up");
    wrap_number_bucket<nb_sp, std::shared_ptr<nb_sp>>(m, "number_bucket_sp");
    wrap_number_bucket<nb_pu, padded_unique_ptr<nb_pu>>(m, "number_bucket_pu");
    wrap_number_bucket<nb_sh, py::smart_holder>(m, "number_bucket_sh");
}
