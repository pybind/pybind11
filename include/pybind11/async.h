#pragma once

#include <memory>
#include <future>
#include <map>
#include <algorithm>
#include <sstream>

#include <chrono>
#include <iostream>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/embed.h"
#include "pybind11/operators.h"


namespace pybind11 {

template<typename ResultType>
class Awaitable : public std::enable_shared_from_this<Awaitable<ResultType>>{
    public:
        Awaitable() {
            this->future = std::future<ResultType>();
        };

        Awaitable(std::future<ResultType>& _future){
            this->future = std::move(_future);
        };

        std::shared_ptr<Awaitable<ResultType>> __iter__() {
            return this->shared_from_this();
        };

        std::shared_ptr<Awaitable<ResultType>> __await__() {
            return this->shared_from_this();
        };

        void __next__() {
            auto status = this->future.wait_for(std::chrono::milliseconds(0));

            if (status == std::future_status::ready) {
                throw pybind11::stop_iteration();
            }
        };

    private:
        std::future<ResultType> future;
};

class async_function : public cpp_function {
    public:
        async_function() = default;
        async_function(std::nullptr_t) { }

        /// Construct a async_function from a vanilla function pointer
        template <typename Return, typename... Args, typename... Extra>
        async_function(Return (*f)(Args...), const Extra&... extra) {
            initialize(f, f, extra...);
        }

        /// Construct a async_function from a lambda function (possibly with internal state)
        template <typename Func, typename... Extra,
                typename = detail::enable_if_t<detail::is_lambda<Func>::value>>
        async_function(Func &&f, const Extra&... extra) {
            initialize(std::forward<Func>(f),
                    (detail::function_signature_t<Func> *) nullptr, extra...);
        }

        /// Construct a async_function from a class method (non-const, no ref-qualifier)
        template <typename Return, typename Class, typename... Arg, typename... Extra>
        async_function(Return (Class::*f)(Arg...), const Extra&... extra) {
            initialize([f](Class *c, Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
                    (Return (*) (Class *, Arg...)) nullptr, extra...);
        }

        /// Construct a async_function from a class method (non-const, lvalue ref-qualifier)
        /// A copy of the overload for non-const functions without explicit ref-qualifier
        /// but with an added `&`.
        template <typename Return, typename Class, typename... Arg, typename... Extra>
        async_function(Return (Class::*f)(Arg...)&, const Extra&... extra) {
            initialize([f](Class *c, Arg... args) -> Return { return (c->*f)(args...); },
                    (Return (*) (Class *, Arg...)) nullptr, extra...);
        }

        /// Construct a async_function from a class method (const, no ref-qualifier)
        template <typename Return, typename Class, typename... Arg, typename... Extra>
        async_function(Return (Class::*f)(Arg...) const, const Extra&... extra) {
            initialize([f](const Class *c, Arg... args) -> Return { return (c->*f)(std::forward<Arg>(args)...); },
                    (Return (*)(const Class *, Arg ...)) nullptr, extra...);
        }

        /// Construct a async_function from a class method (const, lvalue ref-qualifier)
        /// A copy of the overload for const functions without explicit ref-qualifier
        /// but with an added `&`.
        template <typename Return, typename Class, typename... Arg, typename... Extra>
        async_function(Return (Class::*f)(Arg...) const&, const Extra&... extra) {
            initialize([f](const Class *c, Arg... args) -> Return { return (c->*f)(args...); },
                    (Return (*)(const Class *, Arg ...)) nullptr, extra...);
        }

    protected:
        template <typename Func, typename Return, typename... Args, typename... Extra>
        void initialize(Func &&f, Return (*)(Args...), const Extra&... extra) {
            // create a new lambda which spawns an async thread running the original function
            auto proxy = [f](Args... args) -> Awaitable<void>* {
                auto thread_func = std::bind(f, std::forward<Args>(args)...);

                auto future = std::async(std::launch::async, thread_func);
                auto awaitable = new Awaitable<void>(future);

                return awaitable;
            };

            // initialize using the new lambda function
            cpp_function::initialize(
                std::forward<decltype(proxy)>(proxy),
                (detail::function_signature_t<decltype(proxy)> *) nullptr,
                extra...
                );
        }
};

template <typename type_, typename... options>
class class_async : public class_<type_, options...> {

    using type = type_;
    public:
        // using parent constructor
        using class_<type_, options...>::class_;

        template <typename Func, typename... Extra>
        class_async &def_async(const char *name_, Func&& f, const Extra&... extra) {
            async_function cf(method_adaptor<type>(std::forward<Func>(f)), name(name_), is_method(*this),
                            sibling(getattr(*this, name_, none())), extra...);
            add_class_method(*this, name_, cf);
            return *this;
        }

};

}
