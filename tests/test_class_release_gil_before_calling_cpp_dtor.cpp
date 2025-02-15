#include <pybind11/pybind11.h>

#include "pybind11_tests.h"

#include <string>
#include <unordered_map>

namespace pybind11_tests {
namespace class_release_gil_before_calling_cpp_dtor {

using RegistryType = std::unordered_map<std::string, int>;

static RegistryType &PyGILState_Check_Results() {
    static RegistryType singleton; // Local static variables have thread-safe initialization.
    return singleton;
}

template <int> // Using int as a trick to easily generate a series of types.
struct ProbeType {
private:
    std::string unique_key;

public:
    explicit ProbeType(const std::string &unique_key) : unique_key{unique_key} {}

    ~ProbeType() {
        RegistryType &reg = PyGILState_Check_Results();
        assert(reg.count(unique_key) == 0);
        reg[unique_key] = PyGILState_Check();
    }
};

} // namespace class_release_gil_before_calling_cpp_dtor
} // namespace pybind11_tests

TEST_SUBMODULE(class_release_gil_before_calling_cpp_dtor, m) {
    using namespace pybind11_tests::class_release_gil_before_calling_cpp_dtor;

    py::class_<ProbeType<0>>(m, "ProbeType0").def(py::init<std::string>());

    py::class_<ProbeType<1>>(m, "ProbeType1", py::release_gil_before_calling_cpp_dtor())
        .def(py::init<std::string>());

    m.def("PopPyGILState_Check_Result", [](const std::string &unique_key) -> std::string {
        RegistryType &reg = PyGILState_Check_Results();
        if (reg.count(unique_key) == 0) {
            return "MISSING";
        }
        int res = reg[unique_key];
        reg.erase(unique_key);
        return std::to_string(res);
    });
}
