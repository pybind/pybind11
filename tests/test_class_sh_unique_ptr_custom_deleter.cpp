#include "pybind11_tests.h"

#include <memory>

namespace pybind11_tests {
namespace class_sh_unique_ptr_custom_deleter {

// Reduced from a PyCLIF use case in the wild by @wangxf123456.
class Pet {
public:
    using Ptr = std::unique_ptr<Pet, std::function<void(Pet *)>>;

    std::string name;

    static Ptr New(const std::string &name) {
        return Ptr(new Pet(name), std::default_delete<Pet>());
    }

private:
    explicit Pet(const std::string &name) : name(name) {}
};

TEST_SUBMODULE(class_sh_unique_ptr_custom_deleter, m) {
    py::classh<Pet>(m, "Pet").def_readwrite("name", &Pet::name);

    m.def("create", &Pet::New);
}

} // namespace class_sh_unique_ptr_custom_deleter
} // namespace pybind11_tests
