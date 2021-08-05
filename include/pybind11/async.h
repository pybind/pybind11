#pragma once

#include <memory>
#include <future>
#include <chrono>

#include "pybind11/pybind11.h"


namespace py = pybind11;


PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(async)


class StopIteration : public py::stop_iteration {
    public:
        StopIteration(py::object result) : stop_iteration("--"), result(result) {};
        //using py::stop_iteration::stop_iteration;
        /// Set the error using the Python C API
        void set_result(py::object result) {
            this->result = std::move(result);
        }
        void set_error() const override {
            PyErr_SetObject(PyExc_StopIteration, this->result.ptr());
        }
    private:
        py::object result;
};


class Awaitable {
    public:
        Awaitable() {
            this->future = std::future<py::object>();
        };

        Awaitable(std::future<py::object>& _future) {
            this->future = std::move(_future);
        };

        Awaitable* iter() {
            return this;
        };

        Awaitable* await() {
            return this;
        };

        void next() {
            // check future status (zero timeout)
            auto status = this->future.wait_for(std::chrono::milliseconds(0));

            if (status == std::future_status::ready) {
                // future is ready -> raise StopInteration with the future result set
                auto exception = StopIteration(this->future.get());
                //exception.set_result(this->future.get());
                //PyErr_SetObject(PyExc_StopIteration, this->future.get().ptr());

                throw exception;
            }
        };

    private:
        std::future<py::object> future;
};


py::class_<Awaitable> enable_async(py::module m) {
    return py::class_<Awaitable>(m, "Awaitable")
        .def(py::init<>())
        .def("__iter__", &Awaitable::iter)
        .def("__await__", &Awaitable::await)
        .def("__next__", &Awaitable::next);
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
            auto proxy = [f](Args... args) -> Awaitable* {
                auto thread_func = [f](Args... args) {
                    auto result = f(std::forward<Args>(args) ...);

                    py::gil_scoped_acquire gil;

                    auto py_result = py::cast(result);
                    return py_result;
                };
                auto bound_thread_func = std::bind(thread_func, std::forward<Args>(args)...);

                auto future = std::async(std::launch::async, bound_thread_func);
                auto awaitable = new Awaitable(future);

                return awaitable;
            };

            // initialize using the new lambda function
            cpp_function::initialize(
                std::forward<decltype(proxy)>(proxy),
                (detail::function_signature_t<decltype(proxy)> *) nullptr,
                extra...
                );
        }

        template <typename Func, typename... Args, typename... Extra>
        void initialize(Func &&f, void (*)(Args...), const Extra&... extra) {
            // create a new lambda which spawns an async thread running the original function
            auto proxy = [f](Args... args) -> Awaitable* {
                auto thread_func = [f](Args... args) {
                    f(std::forward<Args>(args) ...);

                    py::gil_scoped_acquire gil;

                    auto py_result = py::cast(Py_None);
                    return py_result;
                };
                auto bound_thread_func = std::bind(thread_func, std::forward<Args>(args)...);

                auto future = std::async(std::launch::async, bound_thread_func);
                auto awaitable = new Awaitable(future);

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


PYBIND11_NAMESPACE_END(async)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)