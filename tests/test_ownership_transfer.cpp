/*
    tests/test_ownership_transfer.cpp -- test ownership transfer semantics.

    Copyright (c) 2017 Eric Cousineau <eric.cousineau@tri.global>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#if defined(_MSC_VER) && _MSC_VER < 1910
#  pragma warning(disable: 4702) // unreachable code in system header
#endif

#include <memory>
#include "pybind11_tests.h"
#include "object.h"

enum Label : int {
  BaseBadLabel,
  ChildBadLabel,
  BaseLabel,
  ChildLabel,

  BaseBadUniqueLabel,
  ChildBadUniqueLabel,
  BaseUniqueLabel,
  ChildUniqueLabel,
};

// For attaching instances of `ConstructorStats`.
template <int label>
class Stats {};


template <int label>
class DefineBase {
 public:
  DefineBase(int value)
      : value_(value) {
    track_created(this, value);
  }
  // clang does not like having an implicit copy constructor when the
  // class is virtual (and rightly so).
  DefineBase(const DefineBase&) = delete;
  virtual ~DefineBase() {
    track_destroyed(this);
  }
  virtual int value() const { return value_; }
 private:
  int value_{};
};

template <int label>
class DefineBaseContainer {
 public:
  using T = DefineBase<label>;
  DefineBaseContainer(std::shared_ptr<T> obj)
      : obj_(obj) {}
  std::shared_ptr<T> get() const { return obj_; }
  std::shared_ptr<T> release() { return std::move(obj_); }
 private:
  std::shared_ptr<T> obj_;
};

template <int label>
class DefineBaseUniqueContainer {
 public:
  using T = DefineBase<label>;
  DefineBaseUniqueContainer(std::unique_ptr<T> obj)
      : obj_(std::move(obj)) {}
  T* get() const { return obj_.get(); }
  std::unique_ptr<T> release() { return std::move(obj_); }
 private:
  std::unique_ptr<T> obj_;
};

template <int label>
class DefinePyBase : public DefineBase<label> {
 public:
  using BaseT = DefineBase<label>;
  using BaseT::BaseT;
  int value() const override {
    PYBIND11_OVERLOAD(int, BaseT, value);
  }
};

template <int label>
class DefinePyBaseWrapped : public py::wrapper<DefineBase<label>> {
 public:
  using BaseT = py::wrapper<DefineBase<label>>;
  using BaseT::BaseT;
  int value() const override {
    PYBIND11_OVERLOAD(int, BaseT, value);
  }
};

// BaseBad - No wrapper alias.
typedef DefineBase<BaseBadLabel> BaseBad;
typedef DefineBaseContainer<BaseBadLabel> BaseBadContainer;
typedef Stats<ChildBadLabel> ChildBadStats;

// Base - wrapper alias used in pybind definition.
typedef DefineBase<BaseLabel> Base;
typedef DefinePyBase<BaseLabel> PyBase;
typedef DefineBaseContainer<BaseLabel> BaseContainer;
typedef Stats<ChildLabel> ChildStats;

// - Unique Ptr
// BaseBad - No wrapper alias.
typedef DefineBase<BaseBadUniqueLabel> BaseBadUnique;
typedef DefineBaseUniqueContainer<BaseBadUniqueLabel> BaseBadUniqueContainer;
typedef Stats<ChildBadUniqueLabel> ChildBadUniqueStats;

// Base - wrapper alias used directly.
typedef DefineBase<BaseUniqueLabel> BaseUnique;
typedef DefinePyBaseWrapped<BaseUniqueLabel> PyBaseUnique;
typedef DefineBaseUniqueContainer<BaseUniqueLabel> BaseUniqueContainer;
typedef Stats<ChildUniqueLabel> ChildUniqueStats;

class PyInstanceStats {
 public:
  PyInstanceStats(ConstructorStats& cstats, py::handle h)
    : cstats_(cstats),
      h_(h) {}
  void track_created() {
    cstats_.created(h_.ptr());
    cstats_.value(py::str(h_).cast<std::string>());
  }
  void track_destroyed() {
    cstats_.destroyed(h_.ptr());
  }
 private:
  ConstructorStats& cstats_;
  py::handle h_;
};

PyInstanceStats get_instance_cstats(ConstructorStats& cstats, py::handle h) {
  return PyInstanceStats(cstats, h);
}

template <typename C, typename... Args>
using class_shared_ = py::class_<C, Args..., std::shared_ptr<C>>;

template <typename... Args>
using class_unique_ = py::class_<Args...>;

TEST_SUBMODULE(ownership_transfer, m) {
  // No alias - will not have lifetime extended.
  class_shared_<BaseBad>(m, "BaseBad")
      .def(py::init<int>())
      .def("value", &BaseBad::value);
  class_shared_<BaseBadContainer>(m, "BaseBadContainer")
      .def(py::init<std::shared_ptr<BaseBad>>())
      .def("get", &BaseBadContainer::get)
      .def("release", &BaseBadContainer::release);
  class_shared_<ChildBadStats>(m, "ChildBadStats");

  // Has alias - will have lifetime extended.
  class_shared_<Base, py::wrapper<PyBase>>(m, "Base")
      .def(py::init<int>())
      // Factory method for alias.
      .def(py::init([]() { return new py::wrapper<PyBase>(10); }))
      .def("value", &Base::value);
  class_shared_<BaseContainer>(m, "BaseContainer")
      .def(py::init<std::shared_ptr<Base>>())
      .def("get", &BaseContainer::get)
      .def("release", &BaseContainer::release);
  class_shared_<ChildStats>(m, "ChildStats");

  class_unique_<BaseBadUnique>(m, "BaseBadUnique")
      .def(py::init<int>())
      .def("value", &BaseBadUnique::value);
  class_unique_<BaseBadUniqueContainer>(m, "BaseBadUniqueContainer")
      .def(py::init<std::unique_ptr<BaseBadUnique>>())
      .def("get", &BaseBadUniqueContainer::get)
      .def("release", &BaseBadUniqueContainer::release);
  class_unique_<ChildBadUniqueStats>(m, "ChildBadUniqueStats");

  class_unique_<BaseUnique, PyBaseUnique>(m, "BaseUnique")
      .def(py::init<int>())
      // Factory method.
      .def(py::init([]() { return new PyBaseUnique(10); }))
      .def("value", &BaseUnique::value);
  class_unique_<BaseUniqueContainer>(m, "BaseUniqueContainer")
      .def(py::init<std::unique_ptr<BaseUnique>>())
      .def("get", &BaseUniqueContainer::get)
      .def("release", &BaseUniqueContainer::release);
  class_unique_<ChildUniqueStats>(m, "ChildUniqueStats");

  class_shared_<PyInstanceStats>(m, "InstanceStats")
      .def(py::init<ConstructorStats&, py::handle>())
      .def("track_created", &PyInstanceStats::track_created)
      .def("track_destroyed", &PyInstanceStats::track_destroyed);
  m.def("get_instance_cstats", &get_instance_cstats);
}
