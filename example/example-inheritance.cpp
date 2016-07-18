/*
    example/example-inheritance.cpp -- inheritance, automatic upcasting for polymorphic types

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

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
    void bark() const { std::cout << "Woof!" << std::endl; }
};

class Rabbit : public Pet {
public:
    Rabbit(const std::string &name) : Pet(name, "parrot") {}
};

void pet_print(const Pet &pet) {
    std::cout << pet.name() + " is a " + pet.species() << std::endl;
}

void dog_bark(const Dog &dog) {
    dog.bark();
}


struct BaseClass { virtual ~BaseClass() {} };
struct DerivedClass1 : BaseClass { };
struct DerivedClass2 : BaseClass { };

void init_ex_inheritance(py::module &m) {
    py::class_<Pet> pet_class(m, "Pet");
    pet_class
        .def(py::init<std::string, std::string>())
        .def("name", &Pet::name)
        .def("species", &Pet::species);

    /* One way of declaring a subclass relationship: reference parent's class_ object */
    py::class_<Dog>(m, "Dog", pet_class)
        .def(py::init<std::string>());

    /* Another way of declaring a subclass relationship: reference parent's C++ type */
    py::class_<Rabbit>(m, "Rabbit", py::base<Pet>())
        .def(py::init<std::string>());

    m.def("pet_print", pet_print);
    m.def("dog_bark", dog_bark);

    py::class_<BaseClass>(m, "BaseClass").def(py::init<>());
    py::class_<DerivedClass1>(m, "DerivedClass1").def(py::init<>());
    py::class_<DerivedClass2>(m, "DerivedClass2").def(py::init<>());

    m.def("return_class_1", []() -> BaseClass* { return new DerivedClass1(); });
    m.def("return_class_2", []() -> BaseClass* { return new DerivedClass2(); });
    m.def("return_none", []() -> BaseClass* { return nullptr; });
}
