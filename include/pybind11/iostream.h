#pragma once

#include "pybind11.h"

#include <streambuf>
#include <ostream>
#include <string>
#include <memory>

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

NAMESPACE_BEGIN(detail)

class pythonbuf : public std::streambuf {
private:
    typedef std::streambuf::traits_type traits_type;
    char          d_buffer[1024];

    object pyostream;

    int overflow(int c) {
        if (!traits_type::eq_int_type(c, traits_type::eof())) {
            *this->pptr() = traits_type::to_char_type(c);
            this->pbump(1);
        }
        return this->sync() ? traits_type::not_eof(c) : traits_type::eof();
    }

    int sync() {
        if (this->pbase() != this->pptr()) {
            std::string line(this->pbase(), this->pptr());

            auto write = pyostream.attr("write");
            write(line);
            pyostream.attr("flush")();

            this->setp(this->pbase(), this->epptr());
        }
        return 0;
    }
public:
    pythonbuf(object pyostream)
        : pyostream(pyostream) {
        this->setp(this->d_buffer, this->d_buffer + sizeof(this->d_buffer) - 1);
    }
};

class opythonstream : private virtual pythonbuf, public std::ostream {
public:
    opythonstream(object pyostream)
        : pythonbuf(pyostream),
          std::ostream(static_cast<std::streambuf*>(this)) {
        // this->flags(std::ios_base::unitbuf);
    }
};

NAMESPACE_END(detail)

/** \rst
    Scope redirect
    This a move-only guard and only a single instance can exist.

    .. code-block:: cpp

        #include <pybind11/iostream.h>

        int main() {
            py::scoped_output_redirect redirect{};
            std::cout << "Hello, World!"; // Python stdout
        } // <-- return std::cout to normal

    You can use py::ios::stderr as the constructor argument to gaurd stderr instead.
 \endrst */

class scoped_output_redirect {
    std::streambuf * old {nullptr};
    std::ostream& costream;
    detail::opythonstream buffer;

public:
    scoped_output_redirect( std::ostream& costream, object pyostream) : costream(costream), buffer(pyostream) {
        old = costream.rdbuf(buffer.rdbuf());
    }

    ~scoped_output_redirect() {
        costream.rdbuf(old);
    }

    scoped_output_redirect(const scoped_output_redirect &) = delete;
    scoped_output_redirect(scoped_output_redirect &&other) = default;
    scoped_output_redirect &operator=(const scoped_output_redirect &) = delete;
    scoped_output_redirect &operator=(scoped_output_redirect &&) = delete;
};


NAMESPACE_END(PYBIND11_NAMESPACE)
