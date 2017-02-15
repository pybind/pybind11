/*
    tests/test_inheritance.cpp -- inheritance, automatic upcasting for polymorphic types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

class Pet {
public:
    Pet(const std::string &name, const std::string &species)
        : m_name(name), m_species(species) {}
    std::string name() const { return m_name; }
    std::string species() const { return m_species; }
private:
    std::string m_name;
    std::string m_species;
};

class Dog : public Pet {
public:
    Dog(const std::string &name) : Pet(name, "dog") {}
    std::string bark() const { return "Woof!"; }
};

class Rabbit : public Pet {
public:
    Rabbit(const std::string &name) : Pet(name, "parrot") {}
};

class Hamster : public Pet {
public:
    Hamster(const std::string &name) : Pet(name, "rodent") {}
};

class Chimera : public Pet {
    Chimera() : Pet("Kimmy", "chimera") {}
};

std::string pet_name_species(const Pet &pet) {
    return pet.name() + " is a " + pet.species();
}

std::string dog_bark(const Dog &dog) {
    return dog.bark();
}


struct BaseClass { virtual ~BaseClass() {} };
struct DerivedClass1 : BaseClass { };
struct DerivedClass2 : BaseClass { };

struct MismatchBase1 { };
struct MismatchDerived1 : MismatchBase1 { };

struct MismatchBase2 { };
struct MismatchDerived2 : MismatchBase2 { };

test_initializer inheritance([](py::module &m) {
    py::class_<Pet> pet_class(m, "Pet");
    pet_class
        .def(py::init<std::string, std::string>())
        .def("name", &Pet::name)
        .def("species", &Pet::species);

    /* One way of declaring a subclass relationship: reference parent's class_ object */
    py::class_<Dog>(m, "Dog", pet_class)
        .def(py::init<std::string>());

    /* Another way of declaring a subclass relationship: reference parent's C++ type */
    py::class_<Rabbit, Pet>(m, "Rabbit")
        .def(py::init<std::string>());

    /* And another: list parent in class template arguments */
    py::class_<Hamster, Pet>(m, "Hamster")
        .def(py::init<std::string>());

    py::class_<Chimera, Pet>(m, "Chimera");

    m.def("pet_name_species", pet_name_species);
    m.def("dog_bark", dog_bark);

    py::class_<BaseClass>(m, "BaseClass").def(py::init<>());
    py::class_<DerivedClass1>(m, "DerivedClass1").def(py::init<>());
    py::class_<DerivedClass2>(m, "DerivedClass2").def(py::init<>());

    m.def("return_class_1", []() -> BaseClass* { return new DerivedClass1(); });
    m.def("return_class_2", []() -> BaseClass* { return new DerivedClass2(); });
    m.def("return_class_n", [](int n) -> BaseClass* {
        if (n == 1) return new DerivedClass1();
        if (n == 2) return new DerivedClass2();
        return new BaseClass();
    });
    m.def("return_none", []() -> BaseClass* { return nullptr; });

    m.def("test_isinstance", [](py::list l) {
        struct Unregistered { }; // checks missing type_info code path

        return py::make_tuple(
            py::isinstance<py::tuple>(l[0]),
            py::isinstance<py::dict>(l[1]),
            py::isinstance<Pet>(l[2]),
            py::isinstance<Pet>(l[3]),
            py::isinstance<Dog>(l[4]),
            py::isinstance<Rabbit>(l[5]),
            py::isinstance<Unregistered>(l[6])
        );
    });

    m.def("test_mismatched_holder_type_1", []() {
        auto m = py::module::import("__main__");
        py::class_<MismatchBase1, std::shared_ptr<MismatchBase1>>(m, "MismatchBase1");
        py::class_<MismatchDerived1, MismatchBase1>(m, "MismatchDerived1");
    });
    m.def("test_mismatched_holder_type_2", []() {
        auto m = py::module::import("__main__");
        py::class_<MismatchBase2>(m, "MismatchBase2");
        py::class_<MismatchDerived2, std::shared_ptr<MismatchDerived2>, MismatchBase2>(m, "MismatchDerived2");
    });
});
