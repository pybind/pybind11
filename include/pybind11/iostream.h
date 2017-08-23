#pragma once

/*
    pybind11/iostream -- Tools to assist with redirecting cout and cerr to Python

    Copyright (c) 2017 Henry F. Schreiner

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11.h"

#include <streambuf>
#include <ostream>
#include <string>
#include <memory>
#include <iostream>

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
NAMESPACE_BEGIN(detail)

// Buffer that writes to Python instead of C++
class pythonbuf : public std::streambuf {
private:
    using traits_type = std::streambuf::traits_type;
    char          d_buffer[1024];

    object pywrite;
    object pyflush;

    int overflow(int c) {
        if (!traits_type::eq_int_type(c, traits_type::eof())) {
            *this->pptr() = traits_type::to_char_type(c);
            this->pbump(1);
        }
        return this->sync() ? traits_type::not_eof(c) : traits_type::eof();
    }

    int sync() {
        if (this->pbase() != this->pptr()) {
            // This subtraction cannot be negative, so dropping the sign
            str line(this->pbase(), static_cast<size_t>(this->pptr() - this->pbase()));

            pywrite(line);
            pyflush();

            this->setp(this->pbase(), this->epptr());
        }
        return 0;
    }
public:
    pythonbuf(object pyostream)
        : pywrite(pyostream.attr("write")),
          pyflush(pyostream.attr("flush")) {
        this->setp(this->d_buffer, this->d_buffer + sizeof(this->d_buffer) - 1);
    }
};

NAMESPACE_END(detail)


/** \rst
    Scoped ostream output redirect
    This a move-only guard that redirects output.

    .. code-block:: cpp

        #include <pybind11/iostream.h>

        int main() {
            py::scoped_ostream_redirect output;
            std::cout << "Hello, World!"; // Python stdout
        } // <-- return std::cout to normal

    You can explicitly pass the c++ stream and the python object, for example to gaurd stderr instead.

    .. code-block:: cpp
        int main() {
            py::scoped_ostream_redirect output{std::cerr, py::module::import("sys").attr("stderr")};
            std::cerr << "Hello, World!";
        }
 \endrst */

class scoped_ostream_redirect {
    std::streambuf * old {nullptr};
    std::ostream& costream;
    detail::pythonbuf buffer;

public:
    scoped_ostream_redirect(
            std::ostream& costream = std::cout,
            object pyostream = module::import("sys").attr("stdout") )
        : costream(costream), buffer(pyostream) {
        old = costream.rdbuf(&buffer);
    }

    ~scoped_ostream_redirect() {
        costream.rdbuf(old);
        
    }

    scoped_ostream_redirect(const scoped_ostream_redirect &) = delete;
    scoped_ostream_redirect(scoped_ostream_redirect &&other) = default;
    scoped_ostream_redirect &operator=(const scoped_ostream_redirect &) = delete;
    scoped_ostream_redirect &operator=(scoped_ostream_redirect &&) = delete;
};

NAMESPACE_BEGIN(detail)

// Class to redirect output as a context manager. C++ backend.
class OstreamRedirect {
    bool do_stdout_;
    bool do_stderr_;
    std::unique_ptr<scoped_ostream_redirect> redirect_stdout;
    std::unique_ptr<scoped_ostream_redirect> redirect_stderr;

public:

    OstreamRedirect(bool do_stdout = false, bool do_stderr = false)
        : do_stdout_(do_stdout), do_stderr_(do_stderr) {}

    void enter() {
        // If stdout is true, or if both are false
        if (do_stdout_ || (!do_stdout_ && !do_stderr_)) {
            redirect_stdout.reset(
                new scoped_ostream_redirect(
                    std::cout,
                    module::import("sys").attr("stdout")
                )
            );
        }

        if (do_stderr_) {
            redirect_stderr.reset(
                new scoped_ostream_redirect(
                    std::cerr,
                    module::import("sys").attr("stderr")
                )
            );
        }
    }

    void exit() {
        redirect_stdout.reset();
        redirect_stderr.reset();
    }
};

NAMESPACE_END(detail)

/** \rst
    This is a helper function to add a C++ redirect context manager to Python instead of using a C++ guard.
    To use it, add the following to your binding code:

    .. code-block:: cpp

        #include <pybind11/iostream.h>

        ...

        py::add_ostream_redirect(m, "ostream_redirect");

    You now have a python context manager that redirects your output:

    .. code-block:: python

        with m.ostream_redirect():
            m.print_to_cout_function()

    This manager can optionally be told which streams to operate on:

    .. code-block:: python

        with m.ostream_redirect(stdout=true, stderr=true):
            m.noisy_function_with_error_printing()

 \endrst */

class_<detail::OstreamRedirect> add_ostream_redirect(module m, std::string name = "ostream_redirect") {
    return class_<detail::OstreamRedirect>(m, name.c_str())
        .def(init<bool,bool>(), arg("stdout")=false, arg("stderr")=false)
        .def("__enter__", [](detail::OstreamRedirect &self) {
            self.enter();
        })
        .def("__exit__", [](detail::OstreamRedirect &self, args) {
            self.exit();
        });
}

NAMESPACE_END(PYBIND11_NAMESPACE)
