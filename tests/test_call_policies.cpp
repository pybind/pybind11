/*
    tests/test_call_policies.cpp -- keep_alive and call_guard

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

#include <string>
#include <vector>

struct CustomGuard {
    static bool enabled;

    CustomGuard() { enabled = true; }
    ~CustomGuard() { enabled = false; }

    static const char *report_status() { return enabled ? "guarded" : "unguarded"; }
};
bool CustomGuard::enabled = false;

struct DependentGuard {
    static bool enabled;

    DependentGuard() { enabled = CustomGuard::enabled; }
    ~DependentGuard() { enabled = false; }

    static const char *report_status() { return enabled ? "guarded" : "unguarded"; }
};
bool DependentGuard::enabled = false;

struct CallPolicyHooks {};
struct ThrowingCallPolicyPostcall {};

struct CallGuardState {
    explicit CallGuardState(bool reject) : reject(reject) {}

    bool reject;
    bool guarded = false;
    bool gil_held_during_cast = false;
    std::vector<std::string> events;
};

// The caster scopes this thread-local context to one candidate invocation. Each call owns its
// event log, including when the callable runs with the GIL released.
static CallGuardState *&current_call_guard_state() {
    static thread_local CallGuardState *state = nullptr;
    return state;
}

struct CallGuardArgument {
    explicit CallGuardArgument(CallGuardState *state) : state(state) {}
    CallGuardArgument(CallGuardArgument &&other) noexcept : state(other.state) {}
    CallGuardArgument(const CallGuardArgument &) = delete;
    ~CallGuardArgument() {
        state->events.emplace_back(state->guarded ? "destroy:guarded" : "destroy:unguarded");
    }

    CallGuardState *state;
};

struct ConversionGuard {
    CallGuardState &state = *current_call_guard_state();

    ConversionGuard() {
        state.events.emplace_back("guard:enter");
        state.guarded = true;
    }
    ~ConversionGuard() {
        state.guarded = false;
        state.events.emplace_back("guard:exit");
    }
};

struct NativeCallPolicyHooks {};

namespace PYBIND11_NAMESPACE {
namespace detail {
template <>
struct type_caster<CallGuardArgument> {
    static constexpr auto name = const_name("CallGuardState");
    CallGuardState *state = nullptr;
    CallGuardState *previous = nullptr;

    bool load(handle src, bool) {
        if (!isinstance<CallGuardState>(src)) {
            return false;
        }
        state = &pybind11::cast<CallGuardState &>(src);
        previous = current_call_guard_state();
        current_call_guard_state() = state;
        state->events.emplace_back("load");
        return true;
    }

    ~type_caster() {
        if (state != nullptr) {
            current_call_guard_state() = previous;
        }
    }

    explicit operator CallGuardArgument() const {
        state->events.emplace_back(state->guarded ? "cast:guarded" : "cast:unguarded");
#if !defined(PYPY_VERSION) && !defined(GRAALVM_PYTHON)
        auto *tstate = get_thread_state_unchecked();
        state->gil_held_during_cast
            = tstate != nullptr && tstate == PyGILState_GetThisThreadState();
#endif
        if (state->reject) {
            throw reference_cast_error();
        }
        return CallGuardArgument(state);
    }

    template <typename T>
    using cast_op_type = CallGuardArgument;
};

template <>
struct process_attribute<NativeCallPolicyHooks>
    : process_attribute_default<NativeCallPolicyHooks> {
    static void precall(function_call &call) {
        auto &state = pybind11::cast<CallGuardState &>(call.args[0]);
        state.events.emplace_back(state.guarded ? "precall:guarded" : "precall:unguarded");
    }
    static void postcall(function_call &call, handle) {
        auto &state = pybind11::cast<CallGuardState &>(call.args[0]);
        state.events.emplace_back(state.guarded ? "postcall:guarded" : "postcall:unguarded");
    }
};

template <>
struct process_attribute<CallPolicyHooks> : process_attribute_default<CallPolicyHooks> {
    static void precall(function_call &call) {
        reinterpret_borrow<list>(call.args[0]).append("precall");
    }
    static void postcall(function_call &call, handle result) {
        reinterpret_borrow<list>(call.args[0]).append(result ? "postcall" : "postcall:null");
    }
};
template <>
struct process_attribute<ThrowingCallPolicyPostcall>
    : process_attribute_default<ThrowingCallPolicyPostcall> {
    static void postcall(function_call &, handle) { throw std::runtime_error("postcall failed"); }
};
} // namespace detail
} // namespace PYBIND11_NAMESPACE

TEST_SUBMODULE(call_policies, m) {
    // Parent/Child are used in:
    // test_keep_alive_argument, test_keep_alive_return_value, test_alive_gc_derived,
    // test_alive_gc_multi_derived, test_return_none, test_keep_alive_constructor,
    // test_keep_alive_failed_overload, test_keep_alive_error
    class Child {
    public:
        Child() { py::print("Allocating child."); }
        Child(const Child &) = default;
        Child(Child &&) = default;
        ~Child() { py::print("Releasing child."); }
    };
    py::class_<Child>(m, "Child").def(py::init<>());

    class Parent {
    public:
        Parent() { py::print("Allocating parent."); }
        Parent(const Parent &parent) = default;
        ~Parent() { py::print("Releasing parent."); }
        void addChild(Child *) {}
        Child *returnChild() { return new Child(); }
        Child *returnNullChild() { return nullptr; }
        static Child *staticFunction(Parent *) { return new Child(); }
    };
    py::class_<Parent>(m, "Parent")
        .def(py::init<>())
        .def(py::init([](Child *) { return new Parent(); }), py::keep_alive<1, 2>())
        .def(py::init([](Child *, int) { return new Parent(); }), py::keep_alive<1, 2>())
        .def(py::init([](Child *, const std::string &) { return new Parent(); }))
        .def("addChild", &Parent::addChild)
        .def("addChildKeepAlive", &Parent::addChild, py::keep_alive<1, 2>())
        .def("returnChild", &Parent::returnChild)
        .def("returnChildKeepAlive", &Parent::returnChild, py::keep_alive<1, 0>())
        .def("returnNullChildKeepAliveChild", &Parent::returnNullChild, py::keep_alive<1, 0>())
        .def("returnNullChildKeepAliveParent", &Parent::returnNullChild, py::keep_alive<0, 1>())
        .def_static("staticFunction", &Parent::staticFunction, py::keep_alive<1, 0>());

    m.def("free_function", [](Parent *, Child *) {}, py::keep_alive<1, 2>());

    // test_keep_alive_error
    m.def(
        "keep_alive_error_args",
        [](const py::object &, Child *, py::list &events) { events.append("call"); },
        py::keep_alive<1, 2>());
    m.def(
        "keep_alive_error_return",
        [](const py::object &) { return new Child(); },
        py::keep_alive<1, 0>());

    m.def("invalid_arg_index", [] {}, py::keep_alive<0, 1>());

#if !defined(PYPY_VERSION)
    // test_alive_gc
    class ParentGC : public Parent {
    public:
        using Parent::Parent;
    };
    py::class_<ParentGC, Parent>(m, "ParentGC", py::dynamic_attr()).def(py::init<>());
#endif

    // test_call_guard
    m.def("unguarded_call", &CustomGuard::report_status);
    m.def("guarded_call", &CustomGuard::report_status, py::call_guard<CustomGuard>());

    m.def(
        "multiple_guards_correct_order",
        []() {
            return CustomGuard::report_status() + std::string(" & ")
                   + DependentGuard::report_status();
        },
        py::call_guard<CustomGuard, DependentGuard>());

    m.def(
        "multiple_guards_wrong_order",
        []() {
            return DependentGuard::report_status() + std::string(" & ")
                   + CustomGuard::report_status();
        },
        py::call_guard<DependentGuard, CustomGuard>());

    // These bindings intentionally have no keep_alive. Final caster extraction must still
    // finish before constructing the guard, or firing precall for a rejected candidate.
    py::class_<CallGuardState>(m, "CallGuardState")
        .def(py::init<bool>(), py::arg("reject") = false)
        .def_readonly("gil_held_during_cast", &CallGuardState::gil_held_during_cast)
        .def_property_readonly("events", [](const CallGuardState &state) {
            py::list events;
            for (const auto &event : state.events) {
                events.append(event);
            }
            return events;
        });
    auto guarded_conversion_call = [](CallGuardArgument arg) {
        arg.state->events.emplace_back(arg.state->guarded ? "call:guarded" : "call:unguarded");
    };
    m.def("call_guard_cast", guarded_conversion_call, py::call_guard<ConversionGuard>());
    m.def("call_guard_cast_hooks",
          guarded_conversion_call,
          NativeCallPolicyHooks(),
          py::call_guard<ConversionGuard>());

#if !defined(PYPY_VERSION) && !defined(GRAALVM_PYTHON)
    // `py::call_guard<py::gil_scoped_release>()` should work in PyPy/GraalPy as well,
    // but it's unclear how to test it without `PyGILState_GetThisThreadState`.
    auto report_gil_status = []() {
        auto is_gil_held = false;
        if (auto *tstate = py::detail::get_thread_state_unchecked()) {
            is_gil_held = (tstate == PyGILState_GetThisThreadState());
        }

        return is_gil_held ? "GIL held" : "GIL released";
    };

    m.def("with_gil", report_gil_status);
    m.def("without_gil", report_gil_status, py::call_guard<py::gil_scoped_release>());
    m.def(
        "call_policy_hooks_without_gil",
        [report_gil_status](const py::list &) { return report_gil_status(); },
        CallPolicyHooks(),
        py::call_guard<py::gil_scoped_release>());
    m.def(
        "call_guard_cast_without_gil",
        [report_gil_status](CallGuardArgument arg) {
            arg.state->events.emplace_back(arg.state->guarded ? "call:guarded" : "call:unguarded");
            return report_gil_status();
        },
        py::call_guard<ConversionGuard, py::gil_scoped_release>());
#endif

    // test_keep_alive_failed_overload
    // In each overload pair, the first overload rejects the second argument when the second
    // overload is called; its keep_alive must not fire.
    struct KeepAliveOverload {};
    py::class_<KeepAliveOverload>(m, "KeepAliveOverload").def(py::init<>());
    // Return value as nurse.
    m.def(
        "keep_alive_overload",
        [](const KeepAliveOverload &, int) { return KeepAliveOverload(); },
        py::keep_alive<0, 1>());
    m.def(
        "keep_alive_overload",
        [](const KeepAliveOverload &, const std::string &) { return KeepAliveOverload(); },
        py::keep_alive<0, 1>());
    // Return value as patient.
    m.def(
        "keep_alive_overload_reverse",
        [](const KeepAliveOverload &, int) { return KeepAliveOverload(); },
        py::keep_alive<1, 0>());
    m.def(
        "keep_alive_overload_reverse",
        [](const KeepAliveOverload &, const std::string &) { return KeepAliveOverload(); },
        py::keep_alive<1, 0>());
    // Sole candidate.
    m.def(
        "keep_alive_single",
        [](const KeepAliveOverload &, int) { return KeepAliveOverload(); },
        py::keep_alive<0, 1>());
    // Argument-to-argument.
    m.def("keep_alive_overload_args", [](Parent *, Child *, int) {}, py::keep_alive<1, 2>());
    m.def("keep_alive_overload_args", [](Parent *, Child *, const std::string &) {});
    m.def(
        "keep_alive_overload_args_converting",
        [](Parent *, Child *, const std::string &) {},
        py::keep_alive<1, 2>());
    m.def("keep_alive_overload_args_converting", [](Parent *, Child *, double) {});
    // None loads into a registered-type caster, but cannot be passed as a C++ reference.
    m.def(
        "keep_alive_late_reference_failure",
        [](Parent *, Child *, KeepAliveOverload &) {},
        py::keep_alive<1, 2>());

    m.def(
        "call_policy_hooks",
        [](py::list &events, const std::string &) { events.append("call"); },
        CallPolicyHooks());
    m.def(
        "call_policy_hooks",
        [](py::list &events, double) { events.append("call"); },
        CallPolicyHooks());
    m.def(
        "call_policy_hooks_late_reference",
        [](py::list &events, KeepAliveOverload &) { events.append("call"); },
        CallPolicyHooks());
    m.def(
        "call_policy_hooks_throw",
        [](py::list &events) {
            events.append("call");
            throw std::runtime_error("call failed");
        },
        CallPolicyHooks());
    m.def(
        "call_policy_hooks_throw_postcall",
        [](py::list &events) {
            events.append("call");
            return new Child();
        },
        CallPolicyHooks(),
        ThrowingCallPolicyPostcall());

    // test_keep_alive_failed_return_conversion
    struct UnregisteredType {};
    m.def(
        "keep_alive_unregistered_return",
        [](const KeepAliveOverload &) {
            static UnregisteredType unregistered;
            return &unregistered;
        },
        py::keep_alive<0, 1>(),
        py::return_value_policy::reference);
    m.def(
        "keep_alive_unregistered_return_reverse",
        [](const KeepAliveOverload &) {
            static UnregisteredType unregistered;
            return &unregistered;
        },
        py::keep_alive<1, 0>(),
        py::return_value_policy::reference);
    m.def(
        "call_policy_hooks_unregistered_return",
        [](py::list &events) {
            events.append("call");
            static UnregisteredType unregistered;
            return &unregistered;
        },
        CallPolicyHooks(),
        py::return_value_policy::reference);
}
