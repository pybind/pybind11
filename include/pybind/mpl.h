/*
    pybind/mpl.h: Simple library for type manipulation and template metaprogramming

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#if !defined(__PYBIND_MPL_H)
#define __PYBIND_MPL_H

#include "common.h"
#include <tuple>

NAMESPACE_BEGIN(pybind)
NAMESPACE_BEGIN(mpl)

/// Index sequence for convenient template metaprogramming involving tuples
template<size_t ...> struct index_sequence  { };
template<size_t N, size_t ...S> struct make_index_sequence : make_index_sequence <N - 1, N - 1, S...> { };
template<size_t ...S> struct make_index_sequence <0, S...> { typedef index_sequence<S...> type; };

/// Helper template to strip away type modifiers
template <typename T> struct normalize_type                       { typedef T type; };
template <typename T> struct normalize_type<const T>              { typedef typename normalize_type<T>::type type; };
template <typename T> struct normalize_type<T*>                   { typedef typename normalize_type<T>::type type; };
template <typename T> struct normalize_type<T&>                   { typedef typename normalize_type<T>::type type; };
template <typename T> struct normalize_type<T&&>                  { typedef typename normalize_type<T>::type type; };
template <typename T, size_t N> struct normalize_type<const T[N]> { typedef typename normalize_type<T>::type type; };
template <typename T, size_t N> struct normalize_type<T[N]>       { typedef typename normalize_type<T>::type type; };

NAMESPACE_BEGIN(detail)

/// Strip the class from a method type
template <typename T> struct remove_class {};
template <typename C, typename R, typename... A> struct remove_class<R (C::*)(A...)> { typedef R type(A...); };
template <typename C, typename R, typename... A> struct remove_class<R (C::*)(A...) const> { typedef R type(A...); };

/**
 * \brief Convert a lambda function to a std::function
 * From http://stackoverflow.com/questions/11893141/inferring-the-call-signature-of-a-lambda-or-arbitrary-callable-for-make-functio
 */
template <typename T> struct lambda_signature_impl {
    using type = typename remove_class<
        decltype(&std::remove_reference<T>::type::operator())>::type;
};
template <typename R, typename... A> struct lambda_signature_impl<R    (A...)> { typedef R type(A...); };
template <typename R, typename... A> struct lambda_signature_impl<R (&)(A...)> { typedef R type(A...); };
template <typename R, typename... A> struct lambda_signature_impl<R (*)(A...)> { typedef R type(A...); };
template <typename T> using lambda_signature = typename lambda_signature_impl<T>::type;
template <typename F> using make_function_type = std::function<lambda_signature<F>>;

NAMESPACE_END(detail)

template<typename F> detail::make_function_type<F> make_function(F &&f) {
    return detail::make_function_type<F>(std::forward<F>(f)); }

NAMESPACE_BEGIN(detail)

struct void_type { };

/// Helper functions for calling a function using a tuple argument while dealing with void/non-void return values
template <typename RetType> struct tuple_dispatch {
    typedef RetType return_type;
    template<typename Func, typename Arg, size_t ... S> return_type operator()(const Func &f, Arg && args, index_sequence<S...>) {
        return f(std::get<S>(std::forward<Arg>(args))...);
    }
};

/// Helper functions for calling a function using a tuple argument (special case for void return values)
template <> struct tuple_dispatch<void> {
    typedef void_type return_type;
    template<typename Func, typename Arg, size_t ... S> return_type operator()(const Func &f, Arg &&args, index_sequence<S...>) {
        f(std::get<S>(std::forward<Arg>(args))...);
        return return_type();
    }
};

NAMESPACE_END(detail)

/// For lambda functions delegate to their 'operator()'
template <typename T> struct function_traits : public function_traits<typename detail::make_function_type<T>> { };

/// Type traits for function pointers
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType(*)(Args...)> {
    enum {
        nargs = sizeof...(Args),
        is_method = 0,
        is_const = 0
    };
    typedef std::function<ReturnType (Args...)>    f_type;
    typedef detail::tuple_dispatch<ReturnType>     dispatch_type;
    typedef typename dispatch_type::return_type    return_type;
    typedef std::tuple<Args...>                    args_type;

    template <size_t i> struct arg {
        typedef typename std::tuple_element<i, args_type>::type type;
    };

    static f_type cast(ReturnType (*func)(Args ...)) { return func; }

    static return_type dispatch(const f_type &f, args_type &&args) {
        return dispatch_type()(f, std::move(args),
            typename make_index_sequence<nargs>::type());
    }
};

/// Type traits for ordinary methods
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...)> {
    enum {
        nargs = sizeof...(Args),
        is_method = 1,
        is_const = 0
    };
    typedef std::function<ReturnType(ClassType &, Args...)>  f_type;
    typedef detail::tuple_dispatch<ReturnType>               dispatch_type;
    typedef typename dispatch_type::return_type              return_type;
    typedef std::tuple<ClassType&, Args...>                  args_type;

    template <size_t i> struct arg {
        typedef typename std::tuple_element<i, args_type>::type type;
    };

    static f_type cast(ReturnType (ClassType::*func)(Args ...)) { return std::mem_fn(func); }

    static return_type dispatch(const f_type &f, args_type &&args) {
        return dispatch_type()(f, std::move(args),
            typename make_index_sequence<nargs+1>::type());
    }
};

/// Type traits for const methods
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) const> {
    enum {
        nargs = sizeof...(Args),
        is_method = 1,
        is_const = 1
    };
    typedef std::function<ReturnType (const ClassType &, Args...)>  f_type;
    typedef detail::tuple_dispatch<ReturnType>                      dispatch_type;
    typedef typename dispatch_type::return_type                     return_type;
    typedef std::tuple<const ClassType&, Args...>                   args_type;

    template <size_t i> struct arg {
        typedef typename std::tuple_element<i, args_type>::type type;
    };

    static f_type cast(ReturnType (ClassType::*func)(Args ...) const) {
        return std::mem_fn(func);
    }

    static return_type dispatch(const f_type &f, args_type &&args) {
        return dispatch_type()(f, std::move(args),
            typename make_index_sequence<nargs+1>::type());
    }
};

/// Type traits for std::functions
template <typename ReturnType, typename... Args>
struct function_traits<std::function<ReturnType(Args...)>> {
    enum {
        nargs = sizeof...(Args),
        is_method = 0,
        is_const = 0
    };
    typedef std::function<ReturnType (Args...)>  f_type;
    typedef detail::tuple_dispatch<ReturnType>   dispatch_type;
    typedef typename dispatch_type::return_type  return_type;
    typedef std::tuple<Args...>                  args_type;

    template <size_t i> struct arg {
        typedef typename std::tuple_element<i, args_type>::type type;
    };

    static f_type cast(const f_type &func) { return func; }

    static return_type dispatch(const f_type &f, args_type &&args) {
        return dispatch_type()(f, std::move(args),
            typename make_index_sequence<nargs>::type());
    }
};

NAMESPACE_END(mpl)
NAMESPACE_END(pybind)

#endif /* __PYBIND_MPL_H */
