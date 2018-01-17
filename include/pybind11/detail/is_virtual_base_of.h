/*
    pybind11/detail/virtual_base.h -- provides detail::is_virtual_base_of

    Copyright (c) 2018 Jason Rhinelander <jason@imaginary.ca>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "common.h"

// This metaprogramming template is in its own header because the only way to make gcc not generate
// a warning for the approach used to detect virtual inheritance (which comes from boost) is to tell
// GCC that this is a system header.

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4250) // warning C4250: 'X': inherits 'Y::member' via dominance
#  pragma warning(disable: 4584) // warning C4584: base-class 'X' is already a base-class of 'Y'
#  pragma warning(disable: 4594) // warning C4594: indirect virtual base class is inaccessible
#elif defined(__GNUG__)
// Lie to GCC that this is a system header so as to not generated the unconditional warning for the
// inaccessible base caused by the implementation below when the base class is *not* virtual.
#  pragma GCC system_header
#endif

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename Base, typename Derived, bool is_strict_base> struct is_virtual_base_of_impl {
    struct X : Derived, virtual Base { char data[1024]; };
    struct Y : Derived { char data[1024]; };
    constexpr static bool value = sizeof(X) == sizeof(Y);
};
template <typename Base, typename Derived> struct is_virtual_base_of_impl<Base, Derived, false> : std::false_type {};

/// is_virtual_base_of<Base, Derived>::value is true if and only if Base is a virtual base of Derived.
template <typename Base, typename Derived>
using is_virtual_base_of = bool_constant<is_virtual_base_of_impl<Base, Derived, is_strict_base_of<Base, Derived>::value>::value>;

NAMESPACE_END(detail)
NAMESPACE_END(PYBIND11_NAMESPACE)

#if defined(_MSC_VER)
#  pragma warning(pop)
#endif
