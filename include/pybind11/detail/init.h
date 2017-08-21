/*
    pybind11/detail/init.h: init factory function implementation and support code.

    Copyright (c) 2017 Jason Rhinelander <jason@imaginary.ca>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "class.h"

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
NAMESPACE_BEGIN(detail)
NAMESPACE_BEGIN(initimpl)

inline void no_nullptr(void *ptr) {
    if (!ptr) throw type_error("pybind11::init(): factory function returned nullptr");
}

// Makes sure the `value` for the given value_and_holder is not preallocated (e.g. by a previous
// old-style placement new `__init__` that requires a preallocated, uninitialized value).  If
// preallocated, deallocate.  Returns the (null) value pointer reference ready for allocation.
inline void *&deallocate(value_and_holder &v_h) {
    if (v_h) v_h.type->dealloc(v_h);
    return v_h.value_ptr();
}

PYBIND11_NOINLINE inline value_and_holder load_v_h(handle self_, type_info *tinfo) {
    if (!self_ || !tinfo)
        throw type_error("__init__(self, ...) called with invalid `self` argument");

    auto *inst = reinterpret_cast<instance *>(self_.ptr());
    auto result = inst->get_value_and_holder(tinfo, false);
    if (!result.inst)
        throw type_error("__init__(self, ...) called with invalid `self` argument");

    return result;
}


// Implementing functions for all forms of py::init<...> and py::init(...)
template <typename Class> using Cpp = typename Class::type;
template <typename Class> using Alias = typename Class::type_alias;
template <typename Class> using Holder = typename Class::holder_type;

template <typename Class> using is_alias_constructible = std::is_constructible<Alias<Class>, Cpp<Class> &&>;

// Takes a Cpp pointer and returns true if it actually is a polymorphic Alias instance.
template <typename Class, enable_if_t<Class::has_alias, int> = 0>
bool is_alias(Cpp<Class> *ptr) {
    return dynamic_cast<Alias<Class> *>(ptr) != nullptr;
}
// Failing fallback version of the above for a no-alias class (always returns false)
template <typename /*Class*/>
constexpr bool is_alias(void *) { return false; }

// Attempts to constructs an alias using a `Alias(Cpp &&)` constructor.  This allows types with
// an alias to provide only a single Cpp factory function as long as the Alias can be
// constructed from an rvalue reference of the base Cpp type.  This means that Alias classes
// can, when appropriate, simply define a `Alias(Cpp &&)` constructor rather than needing to
// inherit all the base class constructors.
template <typename Class>
void construct_alias_from_cpp(std::true_type /*is_alias_constructible*/,
                              value_and_holder &v_h, Cpp<Class> &&base) {
    deallocate(v_h) = new Alias<Class>(std::move(base));
}
template <typename Class>
[[noreturn]] void construct_alias_from_cpp(std::false_type /*!is_alias_constructible*/,
                                           value_and_holder &, Cpp<Class> &&) {
    throw type_error("pybind11::init(): unable to convert returned instance to required "
                     "alias class: no `Alias<Class>(Class &&)` constructor available");
}

// Error-generating fallback for factories that don't match one of the below construction
// mechanisms.
template <typename Class>
void construct(...) {
    static_assert(!std::is_same<Class, Class>::value /* always false */,
            "pybind11::init(): init function must return a compatible pointer, "
            "holder, or value");
}

// Pointer return v1: the factory function returns a class pointer for a registered class.
// If we don't need an alias (because this class doesn't have one, or because the final type is
// inherited on the Python side) we can simply take over ownership.  Otherwise we need to try to
// construct an Alias from the returned base instance.
template <typename Class>
void construct(value_and_holder &v_h, Cpp<Class> *ptr, bool need_alias) {
    no_nullptr(ptr);
    if (Class::has_alias && need_alias && !is_alias<Class>(ptr)) {
        // We're going to try to construct an alias by moving the cpp type.  Whether or not
        // that succeeds, we still need to destroy the original cpp pointer (either the
        // moved away leftover, if the alias construction works, or the value itself if we
        // throw an error), but we can't just call `delete ptr`: it might have a special
        // deleter, or might be shared_from_this.  So we construct a holder around it as if
        // it was a normal instance, then steal the holder away into a local variable; thus
        // the holder and destruction happens when we leave the C++ scope, and the holder
        // class gets to handle the destruction however it likes.
        deallocate(v_h) = ptr;
        v_h.set_instance_registered(true); // To prevent init_instance from registering it
        v_h.type->init_instance(v_h.inst, nullptr); // Set up the holder
        Holder<Class> temp_holder(std::move(v_h.holder<Holder<Class>>())); // Steal the holder
        v_h.type->dealloc(v_h); // Destroys the moved-out holder remains, resets value ptr to null
        v_h.set_instance_registered(false);

        construct_alias_from_cpp<Class>(is_alias_constructible<Class>{}, v_h, std::move(*ptr));
    }
    else {
        // Otherwise the type isn't inherited, so we don't need an Alias and can just store the Cpp
        // pointer directory:
        deallocate(v_h) = ptr;
    }
}

// Pointer return v2: a factory that always returns an alias instance ptr.  We simply take over
// ownership of the pointer.
template <typename Class, enable_if_t<Class::has_alias, int> = 0>
void construct(value_and_holder &v_h, Alias<Class> *alias_ptr, bool) {
    no_nullptr(alias_ptr);
    deallocate(v_h) = static_cast<Cpp<Class> *>(alias_ptr);
}

// Holder return: copy its pointer, and move or copy the returned holder into the new instance's
// holder.  This also handles types like std::shared_ptr<T> and std::unique_ptr<T> where T is a
// derived type (through those holder's implicit conversion from derived class holder constructors).
template <typename Class>
void construct(value_and_holder &v_h, Holder<Class> holder, bool need_alias) {
    auto *ptr = holder_helper<Holder<Class>>::get(holder);
    // If we need an alias, check that the held pointer is actually an alias instance
    if (Class::has_alias && need_alias && !is_alias<Class>(ptr))
        throw type_error("pybind11::init(): construction failed: returned holder-wrapped instance "
                         "is not an alias instance");

    deallocate(v_h) = ptr;
    v_h.type->init_instance(v_h.inst, &holder);
}

// return-by-value version 1: returning a cpp class by value.  If the class has an alias and an
// alias is required the alias must have an `Alias(Cpp &&)` constructor so that we can construct
// the alias from the base when needed (i.e. because of Python-side inheritance).  When we don't
// need it, we simply move-construct the cpp value into a new instance.
template <typename Class>
void construct(value_and_holder &v_h, Cpp<Class> &&result, bool need_alias) {
    static_assert(std::is_move_constructible<Cpp<Class>>::value,
        "pybind11::init() return-by-value factory function requires a movable class");
    if (Class::has_alias && need_alias)
        construct_alias_from_cpp<Class>(is_alias_constructible<Class>{}, v_h, std::move(result));
    else
        deallocate(v_h) = new Cpp<Class>(std::move(result));
}

// return-by-value version 2: returning a value of the alias type itself.  We move-construct an
// Alias instance (even if no the python-side inheritance is involved).  The is intended for
// cases where Alias initialization is always desired.
template <typename Class>
void construct(value_and_holder &v_h, Alias<Class> &&result, bool) {
    static_assert(std::is_move_constructible<Alias<Class>>::value,
        "pybind11::init() return-by-alias-value factory function requires a movable alias class");
    deallocate(v_h) = new Alias<Class>(std::move(result));
}

// Implementing class for py::init<...>()
template <typename... Args> struct constructor {
    template <typename Class, typename... Extra, enable_if_t<!Class::has_alias, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        auto *cl_type = get_type_info(typeid(Cpp<Class>));
        cl.def("__init__", [cl_type](handle self_, Args... args) {
            auto v_h = load_v_h(self_, cl_type);
            // If this value is already registered it must mean __init__ is invoked multiple times;
            // we really can't support that in C++, so just ignore the second __init__.
            if (v_h.instance_registered()) return;

            construct<Class>(v_h, new Cpp<Class>{std::forward<Args>(args)...}, false);
        }, extra...);
    }

    template <typename Class, typename... Extra,
              enable_if_t<Class::has_alias &&
                          std::is_constructible<Cpp<Class>, Args...>::value, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        auto *cl_type = get_type_info(typeid(Cpp<Class>));
        cl.def("__init__", [cl_type](handle self_, Args... args) {
            auto v_h = load_v_h(self_, cl_type);
            if (v_h.instance_registered()) return; // Ignore duplicate __init__ calls (see above)

            if (Py_TYPE(v_h.inst) == cl_type->type)
                construct<Class>(v_h, new Cpp<Class>{std::forward<Args>(args)...}, false);
            else
                construct<Class>(v_h, new Alias<Class>{std::forward<Args>(args)...}, true);
        }, extra...);
    }

    template <typename Class, typename... Extra,
              enable_if_t<Class::has_alias &&
                          !std::is_constructible<Cpp<Class>, Args...>::value, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        auto *cl_type = get_type_info(typeid(Cpp<Class>));
        cl.def("__init__", [cl_type](handle self_, Args... args) {
            auto v_h = load_v_h(self_, cl_type);
            if (v_h.instance_registered()) return; // Ignore duplicate __init__ calls (see above)
            construct<Class>(v_h, new Alias<Class>{std::forward<Args>(args)...}, true);
        }, extra...);
    }
};

// Implementing class for py::init_alias<...>()
template <typename... Args> struct alias_constructor {
    template <typename Class, typename... Extra,
              enable_if_t<Class::has_alias && std::is_constructible<Alias<Class>, Args...>::value, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        auto *cl_type = get_type_info(typeid(Cpp<Class>));
        cl.def("__init__", [cl_type](handle self_, Args... args) {
            auto v_h = load_v_h(self_, cl_type);
            if (v_h.instance_registered()) return; // Ignore duplicate __init__ calls (see above)
            construct<Class>(v_h, new Alias<Class>{std::forward<Args>(args)...}, true);
        }, extra...);
    }
};

// Implementation class for py::init(Func) and py::init(Func, AliasFunc)
template <typename CFunc, typename AFuncIn, typename... Args> struct factory {
private:
    using CFuncType = typename std::remove_reference<CFunc>::type;
    using AFunc = conditional_t<std::is_void<AFuncIn>::value, void_type, AFuncIn>;
    using AFuncType = typename std::remove_reference<AFunc>::type;

    CFuncType class_factory;
    AFuncType alias_factory;

public:
    // Constructor with a single function/lambda to call; for classes without aliases or with
    // aliases that can be move constructed from the base.
    factory(CFunc &&f) : class_factory(std::forward<CFunc>(f)) {}

    // Constructor with two functions/lambdas, for a class with distinct class/alias factories: the
    // first is called when an alias is not needed, the second when the alias is needed.  Requires
    // non-void AFunc.
    factory(CFunc &&c, AFunc &&a) :
        class_factory(std::forward<CFunc>(c)), alias_factory(std::forward<AFunc>(a)) {}

    // Add __init__ definition for a class that either has no alias or has no separate alias
    // factory; this always constructs the class itself.  If the class is registered with an alias
    // type and an alias instance is needed (i.e. because the final type is a Python class
    // inheriting from the C++ type) the returned value needs to either already be an alias
    // instance, or the alias needs to be constructible from a `Class &&` argument.
    template <typename Class, typename... Extra,
              enable_if_t<!Class::has_alias || std::is_void<AFuncIn>::value, int> = 0>
    void execute(Class &cl, const Extra&... extra) && {
        auto *cl_type = get_type_info(typeid(Cpp<Class>));
        #if defined(PYBIND11_CPP14)
        cl.def("__init__", [cl_type, func = std::move(class_factory)]
        #else
        CFuncType &func = class_factory;
        cl.def("__init__", [cl_type, func]
        #endif
        (handle self_, Args... args) {
            auto v_h = load_v_h(self_, cl_type);
            // If this value is already registered it must mean __init__ is invoked multiple times;
            // we really can't support that in C++, so just ignore the second __init__.
            if (v_h.instance_registered()) return;

            construct<Class>(v_h, func(std::forward<Args>(args)...), Py_TYPE(v_h.inst) != cl_type->type);
        }, extra...);
    }

    // Add __init__ definition for a class with an alias *and* distinct alias factory; the former is
    // called when the `self` type passed to `__init__` is the direct class (i.e. not inherited), the latter
    // when `self` is a Python-side subtype.
    template <typename Class, typename... Extra,
              enable_if_t<Class::has_alias && !std::is_void<AFuncIn>::value, int> = 0>
    void execute(Class &cl, const Extra&... extra) && {
        auto *cl_type = get_type_info(typeid(Cpp<Class>));

        #if defined(PYBIND11_CPP14)
        cl.def("__init__", [cl_type, class_func = std::move(class_factory), alias_func = std::move(alias_factory)]
        #else
        CFuncType &class_func = class_factory;
        AFuncType &alias_func = alias_factory;
        cl.def("__init__", [cl_type, class_func, alias_func]
        #endif
        (handle self_, Args... args) {
            auto v_h = load_v_h(self_, cl_type);
            if (v_h.instance_registered()) return; // (see comment above)

            if (Py_TYPE(v_h.inst) == cl_type->type)
                // If the instance type equals the registered type we don't have inheritance, so
                // don't need the alias and can construct using the class function:
                construct<Class>(v_h, class_func(std::forward<Args>(args)...), false);
            else
                construct<Class>(v_h, alias_func(std::forward<Args>(args)...), true);
        }, extra...);
    }
};

template <typename Func> using functype =
    conditional_t<std::is_function<remove_reference_t<Func>>::value, remove_reference_t<Func> *,
    conditional_t<is_function_pointer<remove_reference_t<Func>>::value, remove_reference_t<Func>,
    Func>>;

// Helper definition to infer the detail::initimpl::factory template types from a callable object
template <typename Func, typename Return, typename... Args>
factory<functype<Func>, void, Args...> func_decltype(Return (*)(Args...));

// metatemplate that ensures the Class and Alias factories take identical arguments: we need to be
// able to call either one with the given arguments (depending on the final instance type).
template <typename Return1, typename Return2, typename... Args1, typename... Args2>
inline constexpr bool require_matching_arguments(Return1 (*)(Args1...), Return2 (*)(Args2...)) {
    static_assert(sizeof...(Args1) == sizeof...(Args2),
        "pybind11::init(class_factory, alias_factory): class and alias factories must have identical argument signatures");
    static_assert(all_of<std::is_same<Args1, Args2>...>::value,
        "pybind11::init(class_factory, alias_factory): class and alias factories must have identical argument signatures");
    return true;
}

// Unimplemented function provided only for its type signature (via `decltype`), which resolves to
// the appropriate specialization of the above `init` struct with the appropriate function, argument
// and return types.
template <typename CFunc, typename AFunc,
          typename CReturn, typename... CArgs, typename AReturn, typename... AArgs,
          bool = require_matching_arguments((CReturn (*)(CArgs...)) nullptr, (AReturn (*)(AArgs...)) nullptr)>
factory<functype<CFunc>, functype<AFunc>, CArgs...> func_decltype(CReturn (*)(CArgs...), AReturn (*)(AArgs...));

// Resolves to the appropriate specialization of the `pybind11::detail::initimpl::factory<...>` for a
// given init function or pair of class/alias init functions.
template <typename... Func> using factory_t = decltype(func_decltype<Func...>(
    (function_signature_t<Func> *) nullptr...));

NAMESPACE_END(initimpl)
NAMESPACE_END(detail)
NAMESPACE_END(pybind11)
