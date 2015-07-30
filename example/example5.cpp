/*
    example/example5.cpp -- inheritance, callbacks, acquiring and releasing the
    global interpreter lock

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"
#include <pybind/functional.h>


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

void pet_print(const Pet &pet) {
    std::cout << pet.name() + " is a " + pet.species() << std::endl;
}

void dog_bark(const Dog &dog) {
    dog.bark();
}

class Example5  {
public:
    Example5(py::handle self, int state)
        : self(self), state(state) {
        cout << "Constructing Example5.." << endl;
    }

    ~Example5() {
        cout << "Destructing Example5.." << endl;
    }

    void callback(int value) {
		py::gil_scoped_acquire gil;
        cout << "In Example5::callback() " << endl;
        py::object method = self.attr("callback");
        method.call(state, value);
    }
private:
    py::handle self;
    int state;
};

bool test_callback1(py::object func) {
    func.call();
    return false;
}

int test_callback2(py::object func) {
    py::object result = func.call("Hello", true, 5);
    return result.cast<int>();
}

void test_callback3(Example5 *ex, int value) {
	py::gil_scoped_release gil;
    ex->callback(value);
}

void test_callback4(const std::function<int(int)> &func) {
    cout << "func(43) = " << func(43)<< std::endl;
}

std::function<int(int)> test_callback5() {
    return [](int i) { return i+1; };
}

void init_ex5(py::module &m) {
    py::class_<Pet> pet_class(m, "Pet");
    pet_class
        .def(py::init<std::string, std::string>())
        .def("name", &Pet::name)
        .def("species", &Pet::species);

    py::class_<Dog>(m, "Dog", pet_class)
        .def(py::init<std::string>());

    m.def("pet_print", pet_print);
    m.def("dog_bark", dog_bark);

    m.def("test_callback1", &test_callback1);
    m.def("test_callback2", &test_callback2);
    m.def("test_callback3", &test_callback3);
    m.def("test_callback4", &test_callback4);
    m.def("test_callback5", &test_callback5);

    py::class_<Example5>(m, "Example5")
        .def(py::init<py::object, int>());
}
