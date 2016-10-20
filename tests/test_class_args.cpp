/*
    tests/test_class_args.cpp -- tests that various way of defining a class work

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"


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

test_initializer class_args([](py::module &m) {
    // Just test that this compiled okay
    m.def("class_args_noop", []() {});
});
