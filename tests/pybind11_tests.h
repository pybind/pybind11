#pragma once
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

class test_initializer {
    using Initializer = void (*)(py::module &);

public:
    test_initializer(Initializer init);
    test_initializer(const char *submodule_name, Initializer init);
};

#define TEST_SUBMODULE(name, variable)                   \
    void test_submodule_##name(py::module &);            \
    test_initializer name(#name, test_submodule_##name); \
    void test_submodule_##name(py::module &variable)


/// Dummy type which is not exported anywhere -- something to trigger a conversion error
struct UnregisteredType { };

/// A user-defined type which is exported and can be used by any test
class UserType {
public:
    UserType() = default;
    UserType(int i) : i(i) { }

    int value() const { return i; }

private:
    int i = -1;
};

/// Like UserType, but increments `value` on copy for quick reference vs. copy tests
class IncType : public UserType {
public:
    using UserType::UserType;
    IncType() = default;
    IncType(const IncType &other) : IncType(other.value() + 1) { }
    IncType(IncType &&) = delete;
    IncType &operator=(const IncType &) = delete;
    IncType &operator=(IncType &&) = delete;
};
