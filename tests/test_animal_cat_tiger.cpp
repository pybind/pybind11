#include "pybind11_tests.h"

#include <memory>

namespace pybind11_tests {
namespace class_animal {

template <int> // Using int as a trick to easily generate a series of types.
struct Multi {

    class Animal {
    public:
        Animal() = default;
        Animal(const Animal &) = default;
        Animal &operator=(const Animal &) = default;
        virtual std::shared_ptr<Animal> clone() const = 0;
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
        std::shared_ptr<Animal> clone() const override { return std::make_shared<Tiger>(*this); }
    };
};

namespace py = pybind11;

void bind_using_shared_ptr(py::module_ &m) {
    using M = Multi<0>;

    py::class_<M::Animal, std::shared_ptr<M::Animal>>(m, "AnimalSP");

    py::class_<M::Cat, M::Animal, std::shared_ptr<M::Cat>>(m, "CatSP");

    py::class_<M::Tiger, M::Cat, std::shared_ptr<M::Tiger>>(
        m, "TigerSP", py::multiple_inheritance())
        .def(py::init<>())
        .def("clone", &M::Tiger::clone);
}

void bind_using_smart_holder(py::module_ &m) {
    using M = Multi<1>;

    py::class_<M::Animal, py::smart_holder>(m, "AnimalSH");

    py::class_<M::Cat, M::Animal, py::smart_holder>(m, "CatSH");

    py::class_<M::Tiger, M::Cat, py::smart_holder>(m, "TigerSH", py::multiple_inheritance())
        .def(py::init<>())
        .def("clone", &M::Tiger::clone);
}

TEST_SUBMODULE(class_animal, m) {
    bind_using_shared_ptr(m);
    bind_using_smart_holder(m);
}

} // namespace class_animal
} // namespace pybind11_tests
