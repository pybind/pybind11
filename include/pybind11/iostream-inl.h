#include "iostream.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_INLINE int pythonbuf::overflow(int c) {
    if (!traits_type::eq_int_type(c, traits_type::eof())) {
        *pptr() = traits_type::to_char_type(c);
        pbump(1);
    }
    return sync() == 0 ? traits_type::not_eof(c) : traits_type::eof();
}

PYBIND11_INLINE int pythonbuf::sync() {
    if (pbase() != pptr()) {
        // This subtraction cannot be negative, so dropping the sign
        str line(pbase(), static_cast<size_t>(pptr() - pbase()));

        {
            gil_scoped_acquire tmp;
            pywrite(line);
            pyflush();
        }

        setp(pbase(), epptr());
    }
    return 0;
}

PYBIND11_INLINE pythonbuf::pythonbuf(object pyostream, size_t buffer_size)
    : buf_size(buffer_size),
        d_buffer(new char[buf_size]),
        pywrite(pyostream.attr("write")),
        pyflush(pyostream.attr("flush")) {
    setp(d_buffer.get(), d_buffer.get() + buf_size - 1);
}

PYBIND11_INLINE pythonbuf::~pythonbuf() {
    sync();
}

PYBIND11_NAMESPACE_END(detail)

PYBIND11_INLINE scoped_ostream_redirect::scoped_ostream_redirect(
        std::ostream &costream, object pyostream)
    : costream(costream), buffer(pyostream) {
    old = costream.rdbuf(&buffer);
}

PYBIND11_INLINE scoped_ostream_redirect::~scoped_ostream_redirect() {
    costream.rdbuf(old);
}

PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_INLINE void OstreamRedirect::enter() {
    if (do_stdout_)
        redirect_stdout.reset(new scoped_ostream_redirect());
    if (do_stderr_)
        redirect_stderr.reset(new scoped_estream_redirect());
}

PYBIND11_INLINE void OstreamRedirect::exit() {
    redirect_stdout.reset();
    redirect_stderr.reset();
}

PYBIND11_NAMESPACE_END(detail)

PYBIND11_INLINE class_<detail::OstreamRedirect> add_ostream_redirect(module m, std::string name) {
    return class_<detail::OstreamRedirect>(m, name.c_str(), module_local())
        .def(init<bool,bool>(), arg("stdout")=true, arg("stderr")=true)
        .def("__enter__", &detail::OstreamRedirect::enter)
        .def("__exit__", [](detail::OstreamRedirect &self_, args) { self_.exit(); });
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
