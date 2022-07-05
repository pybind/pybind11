#pragma once

#include <pybind11/eval.h>
#include <pybind11/pybind11.h>

#include <cstddef>

namespace py = pybind11;
using namespace pybind11::literals;

class test_initializer {
    using Initializer = void (*)(py::module_ &);

public:
    explicit test_initializer(Initializer init);
    test_initializer(const char *submodule_name, Initializer init);
};

#define TEST_SUBMODULE(name, variable)                                                            \
    void test_submodule_##name(py::module_ &);                                                    \
    test_initializer name(#name, test_submodule_##name);                                          \
    void test_submodule_##name(py::module_ &(variable))

/// Dummy type which is not exported anywhere -- something to trigger a conversion error
struct UnregisteredType {};

/// A user-defined type which is exported and can be used by any test
class UserType {
public:
    UserType() = default;
    explicit UserType(int i) : i(i) {}

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
    IncType(const IncType &other) : IncType(other.value() + 1) {}
    IncType(IncType &&) = delete;
    IncType &operator=(const IncType &) = delete;
    IncType &operator=(IncType &&) = delete;
};

/// A simple union for basic testing
union IntFloat {
    int i;
    float f;
};

/// Custom cast-only type that casts to a string "rvalue" or "lvalue" depending on the cast
/// context. Used to test recursive casters (e.g. std::tuple, stl containers).
struct RValueCaster {};
PYBIND11_NAMESPACE_BEGIN(pybind11)
PYBIND11_NAMESPACE_BEGIN(detail)
template <>
class type_caster<RValueCaster> {
public:
    PYBIND11_TYPE_CASTER(RValueCaster, const_name("RValueCaster"));
    static handle cast(RValueCaster &&, return_value_policy, handle) {
        return py::str("rvalue").release();
    }
    static handle cast(const RValueCaster &, return_value_policy, handle) {
        return py::str("lvalue").release();
    }
};
PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(pybind11)

template <typename F>
void ignoreOldStyleInitWarnings(F &&body) {
    py::exec(R"(
    message = "pybind11-bound class '.+' is using an old-style placement-new '(?:__init__|__setstate__)' which has been deprecated"

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=message, category=FutureWarning)
        body()
    )",
             py::dict(py::arg("body") = py::cpp_function(body)));
}

#ifdef __GNUG__
#    define PYBIND11_NS_VIS_U
#    define PYBIND11_NS_VIS_H __attribute__((visibility("hidden")))
#else
#    define PYBIND11_NS_VIS_U
#    define PYBIND11_NS_VIS_H
#endif

#define PYBIND11_NS_VIS_FUNC                                                                      \
    inline std::ptrdiff_t func(bool get_address) {                                                \
        static std::ptrdiff_t counter = 0;                                                        \
        if (get_address) {                                                                        \
            return reinterpret_cast<std::ptrdiff_t>(&counter);                                    \
        }                                                                                         \
        return counter++;                                                                         \
    }

#define PYBIND11_NS_VIS_DEFS                                                                      \
    m.def("ns_vis_uuu_func", pybind11_ns_vis_uuu::func);                                          \
    m.def("ns_vis_uuh_func", pybind11_ns_vis_uuh::func);                                          \
    m.def("ns_vis_uhu_func", pybind11_ns_vis_uhu::func);                                          \
    m.def("ns_vis_uhh_func", pybind11_ns_vis_uhh::func);                                          \
    m.def("ns_vis_huu_func", pybind11_ns_vis_huu::func);                                          \
    m.def("ns_vis_huh_func", pybind11_ns_vis_huh::func);                                          \
    m.def("ns_vis_hhu_func", pybind11_ns_vis_hhu::func);                                          \
    m.def("ns_vis_hhh_func", pybind11_ns_vis_hhh::func);
