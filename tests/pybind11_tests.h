#pragma once
#include <pybind11/pybind11.h>

#if defined(_MSC_VER) && _MSC_VER < 1910
// We get some really long type names here which causes MSVC 2015 to emit warnings
#  pragma warning(disable: 4503) // warning C4503: decorated name length exceeded, name was truncated
#endif

namespace py = pybind11;
using namespace pybind11::literals;

class test_initializer {
    using Initializer = void (*)(py::module_ &);

public:
    test_initializer(Initializer init);
    test_initializer(const char *submodule_name, Initializer init);
};

#define TEST_SUBMODULE(name, variable)                   \
    void test_submodule_##name(py::module_ &);            \
    test_initializer name(#name, test_submodule_##name); \
    void test_submodule_##name(py::module_ &variable)


/// Dummy type which is not exported anywhere -- something to trigger a conversion error
struct UnregisteredType { };

/// A user-defined type which is exported and can be used by any test
class UserType {
public:
    UserType() = default;
    UserType(int i) : i(i) { }

    int value() const { return i; }
    void set(int set) { i = set; }

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

/// A simple union for basic testing
union IntFloat {
    int i;
    float f;
};

/// Custom cast-only type that casts to a string "rvalue" or "lvalue" depending on the cast context.
/// Used to test recursive casters (e.g. std::tuple, stl containers).
struct RValueCaster {};
PYBIND11_NAMESPACE_BEGIN(pybind11)
PYBIND11_NAMESPACE_BEGIN(detail)
template<> class type_caster<RValueCaster> {
public:
    PYBIND11_TYPE_CASTER(RValueCaster, _("RValueCaster"));
    static handle cast(RValueCaster &&, return_value_policy, handle) { return py::str("rvalue").release(); }
    static handle cast(const RValueCaster &, return_value_policy, handle) { return py::str("lvalue").release(); }
};
PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(pybind11)

/// Simplified ``with warnigns.catch_warnings()`` wrapper
template <typename F>
void ignoreOldStyleInitWarnings(F &&body) {
    auto message = "pybind11-bound class '.+' is using an old-style placement-new '(?:__init__|__setstate__)' which has been deprecated";
    auto category = py::reinterpret_borrow<py::object>(PyExc_FutureWarning);
    auto warnings = py::module_::import("warnings");
    auto context_mgr = warnings.attr("catch_warnings")();
    context_mgr.attr("__enter__")();
    warnings.attr("filterwarnings")("ignore", py::arg("message")=message, py::arg("category")=category);
    body();
    // Exceptions in `body` not handled; see PEP 343 when these would need to be added
    context_mgr.attr("__exit__")(py::none(), py::none(), py::none());
}
