/*
    tests/test_class.cpp -- test py::class_ definitions and basic functionality

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"
#include "constructor_stats.h"

TEST_SUBMODULE(class_, m) {
    // test_instance
    struct NoConstructor {
        static NoConstructor *new_instance() {
            auto *ptr = new NoConstructor();
            print_created(ptr, "via new_instance");
            return ptr;
        }
        ~NoConstructor() { print_destroyed(this); }
    };

    py::class_<NoConstructor>(m, "NoConstructor")
        .def_static("new_instance", &NoConstructor::new_instance, "Return an instance");

    // test_inheritance
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

    /* Constructors are not inherited by default */
    py::class_<Chimera, Pet>(m, "Chimera");

    m.def("pet_name_species", [](const Pet &pet) { return pet.name() + " is a " + pet.species(); });
    m.def("dog_bark", [](const Dog &dog) { return dog.bark(); });

    // test_automatic_upcasting
    struct BaseClass { virtual ~BaseClass() {} };
    struct DerivedClass1 : BaseClass { };
    struct DerivedClass2 : BaseClass { };

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

    // test_isinstance
    m.def("check_instances", [](py::list l) {
        return py::make_tuple(
            py::isinstance<py::tuple>(l[0]),
            py::isinstance<py::dict>(l[1]),
            py::isinstance<Pet>(l[2]),
            py::isinstance<Pet>(l[3]),
            py::isinstance<Dog>(l[4]),
            py::isinstance<Rabbit>(l[5]),
            py::isinstance<UnregisteredType>(l[6])
        );
    });

    // test_mismatched_holder
    struct MismatchBase1 { };
    struct MismatchDerived1 : MismatchBase1 { };

    struct MismatchBase2 { };
    struct MismatchDerived2 : MismatchBase2 { };

    m.def("mismatched_holder_1", []() {
        auto mod = py::module::import("__main__");
        py::class_<MismatchBase1, std::shared_ptr<MismatchBase1>>(mod, "MismatchBase1");
        py::class_<MismatchDerived1, MismatchBase1>(mod, "MismatchDerived1");
    });
    m.def("mismatched_holder_2", []() {
        auto mod = py::module::import("__main__");
        py::class_<MismatchBase2>(mod, "MismatchBase2");
        py::class_<MismatchDerived2, std::shared_ptr<MismatchDerived2>,
                   MismatchBase2>(mod, "MismatchDerived2");
    });

    // test_override_static
    // #511: problem with inheritance + overwritten def_static
    struct MyBase {
        static std::unique_ptr<MyBase> make() {
            return std::unique_ptr<MyBase>(new MyBase());
        }
    };

    struct MyDerived : MyBase {
        static std::unique_ptr<MyDerived> make() {
            return std::unique_ptr<MyDerived>(new MyDerived());
        }
    };

    py::class_<MyBase>(m, "MyBase")
        .def_static("make", &MyBase::make);

    py::class_<MyDerived, MyBase>(m, "MyDerived")
        .def_static("make", &MyDerived::make)
        .def_static("make2", &MyDerived::make);
}

template <int N> class BreaksBase {};
template <int N> class BreaksTramp : public BreaksBase<N> {};
// These should all compile just fine:
typedef py::class_<BreaksBase<1>, std::unique_ptr<BreaksBase<1>>, BreaksTramp<1>> DoesntBreak1;
typedef py::class_<BreaksBase<2>, BreaksTramp<2>, std::unique_ptr<BreaksBase<2>>> DoesntBreak2;
typedef py::class_<BreaksBase<3>, std::unique_ptr<BreaksBase<3>>> DoesntBreak3;
typedef py::class_<BreaksBase<4>, BreaksTramp<4>> DoesntBreak4;
typedef py::class_<BreaksBase<5>> DoesntBreak5;
typedef py::class_<BreaksBase<6>, std::shared_ptr<BreaksBase<6>>, BreaksTramp<6>> DoesntBreak6;
typedef py::class_<BreaksBase<7>, BreaksTramp<7>, std::shared_ptr<BreaksBase<7>>> DoesntBreak7;
typedef py::class_<BreaksBase<8>, std::shared_ptr<BreaksBase<8>>> DoesntBreak8;
#define CHECK_BASE(N) static_assert(std::is_same<typename DoesntBreak##N::type, BreaksBase<N>>::value, \
        "DoesntBreak" #N " has wrong type!")
CHECK_BASE(1); CHECK_BASE(2); CHECK_BASE(3); CHECK_BASE(4); CHECK_BASE(5); CHECK_BASE(6); CHECK_BASE(7); CHECK_BASE(8);
#define CHECK_ALIAS(N) static_assert(DoesntBreak##N::has_alias && std::is_same<typename DoesntBreak##N::type_alias, BreaksTramp<N>>::value, \
        "DoesntBreak" #N " has wrong type_alias!")
#define CHECK_NOALIAS(N) static_assert(!DoesntBreak##N::has_alias && std::is_void<typename DoesntBreak##N::type_alias>::value, \
        "DoesntBreak" #N " has type alias, but shouldn't!")
CHECK_ALIAS(1); CHECK_ALIAS(2); CHECK_NOALIAS(3); CHECK_ALIAS(4); CHECK_NOALIAS(5); CHECK_ALIAS(6); CHECK_ALIAS(7); CHECK_NOALIAS(8);
#define CHECK_HOLDER(N, TYPE) static_assert(std::is_same<typename DoesntBreak##N::holder_type, std::TYPE##_ptr<BreaksBase<N>>>::value, \
        "DoesntBreak" #N " has wrong holder_type!")
CHECK_HOLDER(1, unique); CHECK_HOLDER(2, unique); CHECK_HOLDER(3, unique); CHECK_HOLDER(4, unique); CHECK_HOLDER(5, unique);
CHECK_HOLDER(6, shared); CHECK_HOLDER(7, shared); CHECK_HOLDER(8, shared);

// There's no nice way to test that these fail because they fail to compile; leave them here,
// though, so that they can be manually tested by uncommenting them (and seeing that compilation
// failures occurs).

// We have to actually look into the type: the typedef alone isn't enough to instantiate the type:
#define CHECK_BROKEN(N) static_assert(std::is_same<typename Breaks##N::type, BreaksBase<-N>>::value, \
        "Breaks1 has wrong type!");

//// Two holder classes:
//typedef py::class_<BreaksBase<-1>, std::unique_ptr<BreaksBase<-1>>, std::unique_ptr<BreaksBase<-1>>> Breaks1;
//CHECK_BROKEN(1);
//// Two aliases:
//typedef py::class_<BreaksBase<-2>, BreaksTramp<-2>, BreaksTramp<-2>> Breaks2;
//CHECK_BROKEN(2);
//// Holder + 2 aliases
//typedef py::class_<BreaksBase<-3>, std::unique_ptr<BreaksBase<-3>>, BreaksTramp<-3>, BreaksTramp<-3>> Breaks3;
//CHECK_BROKEN(3);
//// Alias + 2 holders
//typedef py::class_<BreaksBase<-4>, std::unique_ptr<BreaksBase<-4>>, BreaksTramp<-4>, std::shared_ptr<BreaksBase<-4>>> Breaks4;
//CHECK_BROKEN(4);
//// Invalid option (not a subclass or holder)
//typedef py::class_<BreaksBase<-5>, BreaksTramp<-4>> Breaks5;
//CHECK_BROKEN(5);
//// Invalid option: multiple inheritance not supported:
//template <> struct BreaksBase<-8> : BreaksBase<-6>, BreaksBase<-7> {};
//typedef py::class_<BreaksBase<-8>, BreaksBase<-6>, BreaksBase<-7>> Breaks8;
//CHECK_BROKEN(8);
