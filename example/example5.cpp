/*
    example/example5.cpp -- inheritance, callbacks, acquiring and releasing the
    global interpreter lock

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"
#include <pybind11/functional.h>


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

bool test_callback1(py::object func) {
    func();
    return false;
}

int test_callback2(py::object func) {
    py::object result = func("Hello", 'x', true, 5);
    return result.cast<int>();
}

void test_callback3(const std::function<int(int)> &func) {
    cout << "func(43) = " << func(43)<< std::endl;
}

std::function<int(int)> test_callback4() {
    return [](int i) { return i+1; };
}

void init_ex5(py::module &m) {
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

    m.def("test_callback1", &test_callback1);
    m.def("test_callback2", &test_callback2);
    m.def("test_callback3", &test_callback3);
    m.def("test_callback4", &test_callback4);

    /* Test cleanup of lambda closure */

    struct Payload {
        Payload() {
            std::cout << "Payload constructor" << std::endl;
        }
        ~Payload() {
            std::cout << "Payload destructor" << std::endl;
        }
        Payload(const Payload &) {
            std::cout << "Payload copy constructor" << std::endl;
        }
        Payload(Payload &&) {
            std::cout << "Payload move constructor" << std::endl;
        }
    };

    m.def("test_cleanup", []() -> std::function<void(void)> { 
        Payload p;

        return [p]() {
            /* p should be cleaned up when the returned function is garbage collected */
        };
    });
}
