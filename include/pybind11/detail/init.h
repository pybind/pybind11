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

template <>
class type_caster<value_and_holder> {
public:
    bool load(handle h, bool) {
        value = reinterpret_cast<value_and_holder *>(h.ptr());
        return true;
    }

    template <typename> using cast_op_type = value_and_holder &;
    operator value_and_holder &() { return *value; }
    static PYBIND11_DESCR name() { return type_descr(_<value_and_holder>()); }

private:
    value_and_holder *value = nullptr;
};

NAMESPACE_BEGIN(initimpl)

inline void no_nullptr(void *ptr) {
    if (!ptr) throw type_error("pybind11::init(): factory function returned nullptr");
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
    v_h.value_ptr() = new Alias<Class>(std::move(base));
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
        v_h.value_ptr() = ptr;
        v_h.set_instance_registered(true); // To prevent init_instance from registering it
        v_h.type->init_instance(v_h.inst, nullptr); // Set up the holder
        Holder<Class> temp_holder(std::move(v_h.holder<Holder<Class>>())); // Steal the holder
        v_h.type->dealloc(v_h); // Destroys the moved-out holder remains, resets value ptr to null
        v_h.set_instance_registered(false);

        construct_alias_from_cpp<Class>(is_alias_constructible<Class>{}, v_h, std::move(*ptr));
    } else {
        // Otherwise the type isn't inherited, so we don't need an Alias
        v_h.value_ptr() = ptr;
    }
}

// Pointer return v2: a factory that always returns an alias instance ptr.  We simply take over
// ownership of the pointer.
template <typename Class, enable_if_t<Class::has_alias, int> = 0>
void construct(value_and_holder &v_h, Alias<Class> *alias_ptr, bool) {
    no_nullptr(alias_ptr);
    v_h.value_ptr() = static_cast<Cpp<Class> *>(alias_ptr);
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

    v_h.value_ptr() = ptr;
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
        v_h.value_ptr() = new Cpp<Class>(std::move(result));
}

// return-by-value version 2: returning a value of the alias type itself.  We move-construct an
// Alias instance (even if no the python-side inheritance is involved).  The is intended for
// cases where Alias initialization is always desired.
template <typename Class>
void construct(value_and_holder &v_h, Alias<Class> &&result, bool) {
    static_assert(std::is_move_constructible<Alias<Class>>::value,
        "pybind11::init() return-by-alias-value factory function requires a movable alias class");
    v_h.value_ptr() = new Alias<Class>(std::move(result));
}

// Implementing class for py::init<...>()
template <typename... Args>
struct constructor {
    template <typename Class, typename... Extra, enable_if_t<!Class::has_alias, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        cl.def("__init__", [](value_and_holder &v_h, Args... args) {
            v_h.value_ptr() = new Cpp<Class>{std::forward<Args>(args)...};
        }, is_new_style_constructor(), extra...);
    }

    template <typename Class, typename... Extra,
              enable_if_t<Class::has_alias &&
                          std::is_constructible<Cpp<Class>, Args...>::value, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        cl.def("__init__", [](value_and_holder &v_h, Args... args) {
            if (Py_TYPE(v_h.inst) == v_h.type->type)
                v_h.value_ptr() = new Cpp<Class>{std::forward<Args>(args)...};
            else
                v_h.value_ptr() = new Alias<Class>{std::forward<Args>(args)...};
        }, is_new_style_constructor(), extra...);
    }

    template <typename Class, typename... Extra,
              enable_if_t<Class::has_alias &&
                          !std::is_constructible<Cpp<Class>, Args...>::value, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        cl.def("__init__", [](value_and_holder &v_h, Args... args) {
            v_h.value_ptr() = new Alias<Class>{std::forward<Args>(args)...};
        }, is_new_style_constructor(), extra...);
    }
};

// Implementing class for py::init_alias<...>()
template <typename... Args> struct alias_constructor {
    template <typename Class, typename... Extra,
              enable_if_t<Class::has_alias && std::is_constructible<Alias<Class>, Args...>::value, int> = 0>
    static void execute(Class &cl, const Extra&... extra) {
        cl.def("__init__", [](value_and_holder &v_h, Args... args) {
            v_h.value_ptr() = new Alias<Class>{std::forward<Args>(args)...};
        }, is_new_style_constructor(), extra...);
    }
};

// Implementation class for py::init(Func) and py::init(Func, AliasFunc)
template <typename CFunc, typename AFunc = void_type (*)(),
          typename = function_signature_t<CFunc>, typename = function_signature_t<AFunc>>
struct factory;

// Specialization for py::init(Func)
template <typename Func, typename Return, typename... Args>
struct factory<Func, void_type (*)(), Return(Args...)> {
    remove_reference_t<Func> class_factory;

    factory(Func &&f) : class_factory(std::forward<Func>(f)) { }

    // The given class either has no alias or has no separate alias factory;
    // this always constructs the class itself.  If the class is registered with an alias
    // type and an alias instance is needed (i.e. because the final type is a Python class
    // inheriting from the C++ type) the returned value needs to either already be an alias
    // instance, or the alias needs to be constructible from a `Class &&` argument.
    template <typename Class, typename... Extra>
    void execute(Class &cl, const Extra &...extra) && {
        #if defined(PYBIND11_CPP14)
        cl.def("__init__", [func = std::move(class_factory)]
        #else
        auto &func = class_factory;
        cl.def("__init__", [func]
        #endif
        (value_and_holder &v_h, Args... args) {
            construct<Class>(v_h, func(std::forward<Args>(args)...),
                             Py_TYPE(v_h.inst) != v_h.type->type);
        }, is_new_style_constructor(), extra...);
    }
};

// Specialization for py::init(Func, AliasFunc)
template <typename CFunc, typename AFunc,
          typename CReturn, typename... CArgs, typename AReturn, typename... AArgs>
struct factory<CFunc, AFunc, CReturn(CArgs...), AReturn(AArgs...)> {
    static_assert(sizeof...(CArgs) == sizeof...(AArgs),
                  "pybind11::init(class_factory, alias_factory): class and alias factories "
                  "must have identical argument signatures");
    static_assert(all_of<std::is_same<CArgs, AArgs>...>::value,
                  "pybind11::init(class_factory, alias_factory): class and alias factories "
                  "must have identical argument signatures");

    remove_reference_t<CFunc> class_factory;
    remove_reference_t<AFunc> alias_factory;

    factory(CFunc &&c, AFunc &&a)
        : class_factory(std::forward<CFunc>(c)), alias_factory(std::forward<AFunc>(a)) { }

    // The class factory is called when the `self` type passed to `__init__` is the direct
    // class (i.e. not inherited), the alias factory when `self` is a Python-side subtype.
    template <typename Class, typename... Extra>
    void execute(Class &cl, const Extra&... extra) && {
        static_assert(Class::has_alias, "The two-argument version of `py::init()` can "
                                        "only be used if the class has an alias");
        #if defined(PYBIND11_CPP14)
        cl.def("__init__", [class_func = std::move(class_factory), alias_func = std::move(alias_factory)]
        #else
        auto &class_func = class_factory;
        auto &alias_func = alias_factory;
        cl.def("__init__", [class_func, alias_func]
        #endif
        (value_and_holder &v_h, CArgs... args) {
            if (Py_TYPE(v_h.inst) == v_h.type->type)
                // If the instance type equals the registered type we don't have inheritance, so
                // don't need the alias and can construct using the class function:
                construct<Class>(v_h, class_func(std::forward<CArgs>(args)...), false);
            else
                construct<Class>(v_h, alias_func(std::forward<CArgs>(args)...), true);
        }, is_new_style_constructor(), extra...);
    }
};

NAMESPACE_END(initimpl)
NAMESPACE_END(detail)
NAMESPACE_END(pybind11)
