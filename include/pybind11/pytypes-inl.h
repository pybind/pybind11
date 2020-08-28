#include "pytypes.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_INLINE const handle& handle::inc_ref() const & { Py_XINCREF(m_ptr); return *this; }

PYBIND11_INLINE const handle& handle::dec_ref() const & { Py_XDECREF(m_ptr); return *this; }

PYBIND11_INLINE bool handle::operator==(const handle &h) const { return m_ptr == h.m_ptr; }

PYBIND11_INLINE bool handle::operator!=(const handle &h) const { return m_ptr != h.m_ptr; }

PYBIND11_INLINE bool handle::check() const { return m_ptr != nullptr; }

PYBIND11_INLINE object::object(handle h, bool is_borrowed) : handle(h) { if (is_borrowed) inc_ref(); }

PYBIND11_INLINE object::object(const object &o) : handle(o) { inc_ref(); }

PYBIND11_INLINE object::object(object &&other) noexcept { m_ptr = other.m_ptr; other.m_ptr = nullptr; }

PYBIND11_INLINE object::~object() { dec_ref(); }

PYBIND11_INLINE handle object::release() {
    PyObject *tmp = m_ptr;
    m_ptr = nullptr;
    return handle(tmp);
}

PYBIND11_INLINE object& object::operator=(const object &other) {
    other.inc_ref();
    dec_ref();
    m_ptr = other.m_ptr;
    return *this;
}

PYBIND11_INLINE object& object::operator=(object &&other) noexcept {
    if (this != &other) {
        handle temp(m_ptr);
        m_ptr = other.m_ptr;
        other.m_ptr = nullptr;
        temp.dec_ref();
    }
    return *this;
}

PYBIND11_INLINE error_already_set::error_already_set() : std::runtime_error(detail::error_string()) {
    PyErr_Fetch(&m_type.ptr(), &m_value.ptr(), &m_trace.ptr());
}

PYBIND11_INLINE void error_already_set::restore() { PyErr_Restore(m_type.release().ptr(), m_value.release().ptr(), m_trace.release().ptr()); }

PYBIND11_INLINE void error_already_set::discard_as_unraisable(object err_context) {
    restore();
    PyErr_WriteUnraisable(err_context.ptr());
}

PYBIND11_INLINE void error_already_set::discard_as_unraisable(const char *err_context) {
    discard_as_unraisable(reinterpret_steal<object>(PYBIND11_FROM_STRING(err_context)));
}

PYBIND11_INLINE bool error_already_set::matches(handle exc) const { return PyErr_GivenExceptionMatches(m_type.ptr(), exc.ptr()); }

PYBIND11_INLINE bool isinstance(handle obj, handle type) {
    const auto result = PyObject_IsInstance(obj.ptr(), type.ptr());
    if (result == -1)
        throw error_already_set();
    return result != 0;
}

PYBIND11_INLINE bool hasattr(handle obj, handle name) {
    return PyObject_HasAttr(obj.ptr(), name.ptr()) == 1;
}

PYBIND11_INLINE bool hasattr(handle obj, const char *name) {
    return PyObject_HasAttrString(obj.ptr(), name) == 1;
}

PYBIND11_INLINE void delattr(handle obj, handle name) {
    if (PyObject_DelAttr(obj.ptr(), name.ptr()) != 0) { throw error_already_set(); }
}

PYBIND11_INLINE void delattr(handle obj, const char *name) {
    if (PyObject_DelAttrString(obj.ptr(), name) != 0) { throw error_already_set(); }
}

PYBIND11_INLINE object getattr(handle obj, handle name) {
    PyObject *result = PyObject_GetAttr(obj.ptr(), name.ptr());
    if (!result) { throw error_already_set(); }
    return reinterpret_steal<object>(result);
}

PYBIND11_INLINE object getattr(handle obj, const char *name) {
    PyObject *result = PyObject_GetAttrString(obj.ptr(), name);
    if (!result) { throw error_already_set(); }
    return reinterpret_steal<object>(result);
}

PYBIND11_INLINE object getattr(handle obj, handle name, handle default_) {
    if (PyObject *result = PyObject_GetAttr(obj.ptr(), name.ptr())) {
        return reinterpret_steal<object>(result);
    } else {
        PyErr_Clear();
        return reinterpret_borrow<object>(default_);
    }
}

PYBIND11_INLINE object getattr(handle obj, const char *name, handle default_) {
    if (PyObject *result = PyObject_GetAttrString(obj.ptr(), name)) {
        return reinterpret_steal<object>(result);
    } else {
        PyErr_Clear();
        return reinterpret_borrow<object>(default_);
    }
}

PYBIND11_INLINE void setattr(handle obj, handle name, handle value) {
    if (PyObject_SetAttr(obj.ptr(), name.ptr(), value.ptr()) != 0) { throw error_already_set(); }
}

PYBIND11_INLINE void setattr(handle obj, const char *name, handle value) {
    if (PyObject_SetAttrString(obj.ptr(), name, value.ptr()) != 0) { throw error_already_set(); }
}

PYBIND11_INLINE ssize_t hash(handle obj) {
    auto h = PyObject_Hash(obj.ptr());
    if (h == -1) { throw error_already_set(); }
    return h;
}

PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_INLINE handle get_function(handle value) {
    if (value) {
#if PY_MAJOR_VERSION >= 3
        if (PyInstanceMethod_Check(value.ptr()))
            value = PyInstanceMethod_GET_FUNCTION(value.ptr());
        else
#endif
        if (PyMethod_Check(value.ptr()))
            value = PyMethod_GET_FUNCTION(value.ptr());
    }
    return value;
}

PYBIND11_NAMESPACE_BEGIN(accessor_policies)

PYBIND11_INLINE object obj_attr::get(handle obj, handle key) { return getattr(obj, key); }

PYBIND11_INLINE void obj_attr::set(handle obj, handle key, handle val) { setattr(obj, key, val); }

PYBIND11_INLINE object str_attr::get(handle obj, const char *key) { return getattr(obj, key); }

PYBIND11_INLINE void str_attr::set(handle obj, const char *key, handle val) { setattr(obj, key, val); }

PYBIND11_INLINE object generic_item::get(handle obj, handle key) {
    PyObject *result = PyObject_GetItem(obj.ptr(), key.ptr());
    if (!result) { throw error_already_set(); }
    return reinterpret_steal<object>(result);
}

PYBIND11_INLINE void generic_item::set(handle obj, handle key, handle val) {
    if (PyObject_SetItem(obj.ptr(), key.ptr(), val.ptr()) != 0) { throw error_already_set(); }
}

PYBIND11_INLINE object sequence_item::get(handle obj, size_t index) {
    PyObject *result = PySequence_GetItem(obj.ptr(), static_cast<ssize_t>(index));
    if (!result) { throw error_already_set(); }
    return reinterpret_steal<object>(result);
}

PYBIND11_INLINE void sequence_item::set(handle obj, size_t index, handle val) {
    // PySequence_SetItem does not steal a reference to 'val'
    if (PySequence_SetItem(obj.ptr(), static_cast<ssize_t>(index), val.ptr()) != 0) {
        throw error_already_set();
    }
}

PYBIND11_INLINE object list_item::get(handle obj, size_t index) {
    PyObject *result = PyList_GetItem(obj.ptr(), static_cast<ssize_t>(index));
    if (!result) { throw error_already_set(); }
    return reinterpret_borrow<object>(result);
}

PYBIND11_INLINE void list_item::set(handle obj, size_t index, handle val) {
    // PyList_SetItem steals a reference to 'val'
    if (PyList_SetItem(obj.ptr(), static_cast<ssize_t>(index), val.inc_ref().ptr()) != 0) {
        throw error_already_set();
    }
}

PYBIND11_INLINE object tuple_item::get(handle obj, size_t index) {
    PyObject *result = PyTuple_GetItem(obj.ptr(), static_cast<ssize_t>(index));
    if (!result) { throw error_already_set(); }
    return reinterpret_borrow<object>(result);
}

PYBIND11_INLINE void tuple_item::set(handle obj, size_t index, handle val) {
    // PyTuple_SetItem steals a reference to 'val'
    if (PyTuple_SetItem(obj.ptr(), static_cast<ssize_t>(index), val.inc_ref().ptr()) != 0) {
        throw error_already_set();
    }
}

PYBIND11_NAMESPACE_END(accessor_policies)

PYBIND11_NAMESPACE_BEGIN(iterator_policies)

PYBIND11_INLINE sequence_fast_readonly::sequence_fast_readonly(handle obj, ssize_t n) : ptr(PySequence_Fast_ITEMS(obj.ptr()) + n) { }

PYBIND11_INLINE sequence_fast_readonly::reference sequence_fast_readonly::dereference() const { return *ptr; }

PYBIND11_INLINE void sequence_fast_readonly::increment() { ++ptr; }

PYBIND11_INLINE void sequence_fast_readonly::decrement() { --ptr; }

PYBIND11_INLINE void sequence_fast_readonly::advance(ssize_t n) { ptr += n; }

PYBIND11_INLINE bool sequence_fast_readonly::equal(const sequence_fast_readonly &b) const { return ptr == b.ptr; }

PYBIND11_INLINE ssize_t sequence_fast_readonly::distance_to(const sequence_fast_readonly &b) const { return ptr - b.ptr; }

PYBIND11_INLINE sequence_slow_readwrite::reference sequence_slow_readwrite::dereference() const { return {obj, static_cast<size_t>(index)}; }

PYBIND11_INLINE void sequence_slow_readwrite::increment() { ++index; }

PYBIND11_INLINE void sequence_slow_readwrite::decrement() { --index; }

PYBIND11_INLINE void sequence_slow_readwrite::advance(ssize_t n) { index += n; }

PYBIND11_INLINE bool sequence_slow_readwrite::equal(const sequence_slow_readwrite &b) const { return index == b.index; }

PYBIND11_INLINE ssize_t sequence_slow_readwrite::distance_to(const sequence_slow_readwrite &b) const { return index - b.index; }

PYBIND11_INLINE dict_readonly::dict_readonly(handle obj, ssize_t pos) : obj(obj), pos(pos) { increment(); }

PYBIND11_INLINE dict_readonly::reference dict_readonly::dereference() const { return {key, value}; }

PYBIND11_INLINE void dict_readonly::increment() { if (!PyDict_Next(obj.ptr(), &pos, &key, &value)) { pos = -1; } }

PYBIND11_INLINE bool dict_readonly::equal(const dict_readonly &b) const { return pos == b.pos; }

PYBIND11_NAMESPACE_END(iterator_policies)

PYBIND11_INLINE bool PyIterable_Check(PyObject *obj) {
    PyObject *iter = PyObject_GetIter(obj);
    if (iter) {
        Py_DECREF(iter);
        return true;
    } else {
        PyErr_Clear();
        return false;
    }
}

PYBIND11_INLINE bool PyNone_Check(PyObject *o) { return o == Py_None; }

PYBIND11_INLINE bool PyEllipsis_Check(PyObject *o) { return o == Py_Ellipsis; }

PYBIND11_INLINE bool PyUnicode_Check_Permissive(PyObject *o) { return PyUnicode_Check(o) || PYBIND11_BYTES_CHECK(o); }

PYBIND11_INLINE bool PyStaticMethod_Check(PyObject *o) { return o->ob_type == &PyStaticMethod_Type; }

PYBIND11_INLINE kwargs_proxy args_proxy::operator*() const { return kwargs_proxy(*this); }

PYBIND11_NAMESPACE_END(detail)

PYBIND11_INLINE iterator& iterator::operator++() {
    advance();
    return *this;
}

PYBIND11_INLINE iterator iterator::operator++(int) {
    auto rv = *this;
    advance();
    return rv;
}

PYBIND11_INLINE iterator::reference iterator::operator*() const {
    if (m_ptr && !value.ptr()) {
        auto& self = const_cast<iterator &>(*this);
        self.advance();
    }
    return value;
}

PYBIND11_INLINE iterator::pointer iterator::operator->() const { operator*(); return &value; }

PYBIND11_INLINE iterator iterator::sentinel() { return {}; }

PYBIND11_INLINE bool operator==(const iterator &a, const iterator &b) { return a->ptr() == b->ptr(); }

PYBIND11_INLINE bool operator!=(const iterator &a, const iterator &b) { return a->ptr() != b->ptr(); }

PYBIND11_INLINE void iterator::advance() {
    value = reinterpret_steal<object>(PyIter_Next(m_ptr));
    if (PyErr_Occurred()) { throw error_already_set(); }
}

PYBIND11_INLINE str::str(const char *c, size_t n)
    : object(PyUnicode_FromStringAndSize(c, (ssize_t) n), stolen_t{}) {
    if (!m_ptr) pybind11_fail("Could not allocate string object!");
}

PYBIND11_INLINE str::str(const char *c)
    : object(PyUnicode_FromString(c), stolen_t{}) {
    if (!m_ptr) pybind11_fail("Could not allocate string object!");
}

PYBIND11_INLINE str::str(const std::string &s) : str(s.data(), s.size()) { }

PYBIND11_INLINE str::str(handle h) : object(raw_str(h.ptr()), stolen_t{}) { }

PYBIND11_INLINE str::operator std::string() const {
    object temp = *this;
    if (PyUnicode_Check(m_ptr)) {
        temp = reinterpret_steal<object>(PyUnicode_AsUTF8String(m_ptr));
        if (!temp)
            pybind11_fail("Unable to extract string contents! (encoding issue)");
    }
    char *buffer;
    ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(temp.ptr(), &buffer, &length))
        pybind11_fail("Unable to extract string contents! (invalid type)");
    return std::string(buffer, (size_t) length);
}

PYBIND11_INLINE PyObject *str::raw_str(PyObject *op) {
    PyObject *str_value = PyObject_Str(op);
    if (!str_value) throw error_already_set();
#if PY_MAJOR_VERSION < 3
    PyObject *unicode = PyUnicode_FromEncodedObject(str_value, "utf-8", nullptr);
    Py_XDECREF(str_value); str_value = unicode;
#endif
    return str_value;
}

inline namespace literals {
PYBIND11_INLINE str operator"" _s(const char *s, size_t size) { return {s, size}; }
}

PYBIND11_INLINE bytes::bytes(const char *c)
    : object(PYBIND11_BYTES_FROM_STRING(c), stolen_t{}) {
    if (!m_ptr) pybind11_fail("Could not allocate bytes object!");
}

PYBIND11_INLINE bytes::bytes(const char *c, size_t n)
    : object(PYBIND11_BYTES_FROM_STRING_AND_SIZE(c, (ssize_t) n), stolen_t{}) {
    if (!m_ptr) pybind11_fail("Could not allocate bytes object!");
}

     // Allow implicit conversion:
PYBIND11_INLINE bytes::bytes(const std::string &s) : bytes(s.data(), s.size()) { }

PYBIND11_INLINE bytes::operator std::string() const {
    char *buffer;
    ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(m_ptr, &buffer, &length))
        pybind11_fail("Unable to extract bytes contents!");
    return std::string(buffer, (size_t) length);
}

// Note: breathe >= 4.17.0 will fail to build docs if the below two constructors
// are included in the doxygen group; close here and reopen after as a workaround

PYBIND11_INLINE bytes::bytes(const pybind11::str &s) {
    object temp = s;
    if (PyUnicode_Check(s.ptr())) {
        temp = reinterpret_steal<object>(PyUnicode_AsUTF8String(s.ptr()));
        if (!temp)
            pybind11_fail("Unable to extract string contents! (encoding issue)");
    }
    char *buffer;
    ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(temp.ptr(), &buffer, &length))
        pybind11_fail("Unable to extract string contents! (invalid type)");
    auto obj = reinterpret_steal<object>(PYBIND11_BYTES_FROM_STRING_AND_SIZE(buffer, length));
    if (!obj)
        pybind11_fail("Could not allocate bytes object!");
    m_ptr = obj.release().ptr();
}

PYBIND11_INLINE str::str(const bytes& b) {
    char *buffer;
    ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(b.ptr(), &buffer, &length))
        pybind11_fail("Unable to extract bytes contents!");
    auto obj = reinterpret_steal<object>(PyUnicode_FromStringAndSize(buffer, (ssize_t) length));
    if (!obj)
        pybind11_fail("Could not allocate string object!");
    m_ptr = obj.release().ptr();
}

PYBIND11_INLINE none::none() : object(Py_None, borrowed_t{}) { }

PYBIND11_INLINE ellipsis::ellipsis() : object(Py_Ellipsis, borrowed_t{}) { }

PYBIND11_INLINE bool_::bool_() : object(Py_False, borrowed_t{}) { }

PYBIND11_INLINE bool_::bool_(bool value) : object(value ? Py_True : Py_False, borrowed_t{}) { }

PYBIND11_INLINE bool_::operator bool() const { return m_ptr && PyLong_AsLong(m_ptr) != 0; }

PYBIND11_INLINE PyObject *bool_::raw_bool(PyObject *op) {
    const auto value = PyObject_IsTrue(op);
    if (value == -1) return nullptr;
    return handle(value ? Py_True : Py_False).inc_ref().ptr();
}

PYBIND11_INLINE int_::int_() : object(PyLong_FromLong(0), stolen_t{}) { }

PYBIND11_INLINE float_::float_(float value) : object(PyFloat_FromDouble((double) value), stolen_t{}) {
    if (!m_ptr) pybind11_fail("Could not allocate float object!");
}

PYBIND11_INLINE float_::float_(double value) : object(PyFloat_FromDouble((double) value), stolen_t{}) {
    if (!m_ptr) pybind11_fail("Could not allocate float object!");
}

PYBIND11_INLINE float_::operator float() const { return (float) PyFloat_AsDouble(m_ptr); }

PYBIND11_INLINE float_::operator double() const { return (double) PyFloat_AsDouble(m_ptr); }

PYBIND11_INLINE weakref::weakref(handle obj, handle callback)
    : object(PyWeakref_NewRef(obj.ptr(), callback.ptr()), stolen_t{}) {
    if (!m_ptr) pybind11_fail("Could not allocate weak reference!");
}

PYBIND11_INLINE slice::slice(ssize_t start_, ssize_t stop_, ssize_t step_) {
    int_ start(start_), stop(stop_), step(step_);
    m_ptr = PySlice_New(start.ptr(), stop.ptr(), step.ptr());
    if (!m_ptr) pybind11_fail("Could not allocate slice object!");
}

PYBIND11_INLINE bool slice::compute(size_t length, size_t *start, size_t *stop, size_t *step,
                size_t *slicelength) const {
    return PySlice_GetIndicesEx((PYBIND11_SLICE_OBJECT *) m_ptr,
                                (ssize_t) length, (ssize_t *) start,
                                (ssize_t *) stop, (ssize_t *) step,
                                (ssize_t *) slicelength) == 0;
}

PYBIND11_INLINE bool slice::compute(ssize_t length, ssize_t *start, ssize_t *stop, ssize_t *step,
    ssize_t *slicelength) const {
    return PySlice_GetIndicesEx((PYBIND11_SLICE_OBJECT *) m_ptr,
        length, start,
        stop, step,
        slicelength) == 0;
}

PYBIND11_INLINE capsule::capsule(PyObject *ptr, bool is_borrowed) : object(is_borrowed ? object(ptr, borrowed_t{}) : object(ptr, stolen_t{})) { }

PYBIND11_INLINE capsule::capsule(const void *value, const char *name, void (*destructor)(PyObject *))
    : object(PyCapsule_New(const_cast<void *>(value), name, destructor), stolen_t{}) {
    if (!m_ptr)
        pybind11_fail("Could not allocate capsule object!");
}

PYBIND11_INLINE capsule::capsule(const void *value, void (*destruct)(PyObject *))
    : object(PyCapsule_New(const_cast<void*>(value), nullptr, destruct), stolen_t{}) {
    if (!m_ptr)
        pybind11_fail("Could not allocate capsule object!");
}

PYBIND11_INLINE capsule::capsule(const void *value, void (*destructor)(void *)) {
    m_ptr = PyCapsule_New(const_cast<void *>(value), nullptr, [](PyObject *o) {
        auto destructor = reinterpret_cast<void (*)(void *)>(PyCapsule_GetContext(o));
        void *ptr = PyCapsule_GetPointer(o, nullptr);
        destructor(ptr);
    });

    if (!m_ptr)
        pybind11_fail("Could not allocate capsule object!");

    if (PyCapsule_SetContext(m_ptr, (void *) destructor) != 0)
        pybind11_fail("Could not set capsule context!");
}

PYBIND11_INLINE capsule::capsule(void (*destructor)()) {
    m_ptr = PyCapsule_New(reinterpret_cast<void *>(destructor), nullptr, [](PyObject *o) {
        auto destructor = reinterpret_cast<void (*)()>(PyCapsule_GetPointer(o, nullptr));
        destructor();
    });

    if (!m_ptr)
        pybind11_fail("Could not allocate capsule object!");
}

PYBIND11_INLINE const char *capsule::name() const { return PyCapsule_GetName(m_ptr); }

PYBIND11_INLINE tuple::tuple(size_t size) : object(PyTuple_New((ssize_t) size), stolen_t{}) {
    if (!m_ptr) pybind11_fail("Could not allocate tuple object!");
}

PYBIND11_INLINE size_t tuple::size() const { return (size_t) PyTuple_Size(m_ptr); }

PYBIND11_INLINE bool tuple::empty() const { return size() == 0; }

PYBIND11_INLINE detail::tuple_accessor tuple::operator[](size_t index) const { return {*this, index}; }

PYBIND11_INLINE detail::item_accessor tuple::operator[](handle h) const { return object::operator[](h); }

PYBIND11_INLINE detail::tuple_iterator tuple::begin() const { return {*this, 0}; }

PYBIND11_INLINE detail::tuple_iterator tuple::end() const { return {*this, PyTuple_GET_SIZE(m_ptr)}; }

PYBIND11_INLINE dict::dict() : object(PyDict_New(), stolen_t{}) {
    if (!m_ptr) pybind11_fail("Could not allocate dict object!");
}

PYBIND11_INLINE size_t dict::size() const { return (size_t) PyDict_Size(m_ptr); }

PYBIND11_INLINE bool dict::empty() const { return size() == 0; }

PYBIND11_INLINE detail::dict_iterator dict::begin() const { return {*this, 0}; }

PYBIND11_INLINE detail::dict_iterator dict::end() const { return {}; }

PYBIND11_INLINE void dict::clear() const { PyDict_Clear(ptr()); }

PYBIND11_INLINE PyObject *dict::raw_dict(PyObject *op) {
    if (PyDict_Check(op))
        return handle(op).inc_ref().ptr();
    return PyObject_CallFunctionObjArgs((PyObject *) &PyDict_Type, op, nullptr);
}

PYBIND11_INLINE size_t sequence::size() const {
    ssize_t result = PySequence_Size(m_ptr);
    if (result == -1)
        throw error_already_set();
    return (size_t) result;
}

PYBIND11_INLINE bool sequence::empty() const { return size() == 0; }

PYBIND11_INLINE detail::sequence_accessor sequence::operator[](size_t index) const { return {*this, index}; }

PYBIND11_INLINE detail::item_accessor sequence::operator[](handle h) const { return object::operator[](h); }

PYBIND11_INLINE detail::sequence_iterator sequence::begin() const { return {*this, 0}; }

PYBIND11_INLINE detail::sequence_iterator sequence::end() const { return {*this, PySequence_Size(m_ptr)}; }

PYBIND11_INLINE list::list(size_t size) : object(PyList_New((ssize_t) size), stolen_t{}) {
    if (!m_ptr) pybind11_fail("Could not allocate list object!");
}

PYBIND11_INLINE size_t list::size() const { return (size_t) PyList_Size(m_ptr); }

PYBIND11_INLINE bool list::empty() const { return size() == 0; }

PYBIND11_INLINE detail::list_accessor list::operator[](size_t index) const { return {*this, index}; }

PYBIND11_INLINE detail::item_accessor list::operator[](handle h) const { return object::operator[](h); }

PYBIND11_INLINE detail::list_iterator list::begin() const { return {*this, 0}; }

PYBIND11_INLINE detail::list_iterator list::end() const { return {*this, PyList_GET_SIZE(m_ptr)}; }

PYBIND11_INLINE set::set() : object(PySet_New(nullptr), stolen_t{}) {
    if (!m_ptr) pybind11_fail("Could not allocate set object!");
}

PYBIND11_INLINE size_t set::size() const { return (size_t) PySet_Size(m_ptr); }

PYBIND11_INLINE bool set::empty() const { return size() == 0; }

PYBIND11_INLINE handle function::cpp_function() const {
    handle fun = detail::get_function(m_ptr);
    if (fun && PyCFunction_Check(fun.ptr()))
        return fun;
    return handle();
}

PYBIND11_INLINE bool function::is_cpp_function() const { return (bool) cpp_function(); }

PYBIND11_INLINE buffer_info buffer::request(bool writable) const {
    int flags = PyBUF_STRIDES | PyBUF_FORMAT;
    if (writable) flags |= PyBUF_WRITABLE;
    Py_buffer *view = new Py_buffer();
    if (PyObject_GetBuffer(m_ptr, view, flags) != 0) {
        delete view;
        throw error_already_set();
    }
    return buffer_info(view);
}

PYBIND11_INLINE memoryview::memoryview(const buffer_info& info) {
    if (!info.view())
        pybind11_fail("Prohibited to create memoryview without Py_buffer");
    // Note: PyMemoryView_FromBuffer never increments obj reference.
    m_ptr = (info.view()->obj) ?
        PyMemoryView_FromObject(info.view()->obj) :
        PyMemoryView_FromBuffer(info.view());
    if (!m_ptr)
        pybind11_fail("Unable to create memoryview from buffer descriptor");
}

PYBIND11_INLINE memoryview memoryview::from_buffer(
    const void *ptr, ssize_t itemsize, const char *format,
    detail::any_container<ssize_t> shape,
    detail::any_container<ssize_t> strides) {
    return memoryview::from_buffer(
        const_cast<void*>(ptr), itemsize, format, shape, strides, true);
}

#if PY_MAJOR_VERSION >= 3
PYBIND11_INLINE memoryview memoryview::from_memory(void *mem, ssize_t size, bool readonly) {
    PyObject* ptr = PyMemoryView_FromMemory(
        reinterpret_cast<char*>(mem), size,
        (readonly) ? PyBUF_READ : PyBUF_WRITE);
    if (!ptr)
        pybind11_fail("Could not allocate memoryview object!");
    return memoryview(object(ptr, stolen_t{}));
}

PYBIND11_INLINE memoryview memoryview::from_memory(const void *mem, ssize_t size) {
    return memoryview::from_memory(const_cast<void*>(mem), size, true);
}
#endif

#ifndef DOXYGEN_SHOULD_SKIP_THIS
PYBIND11_INLINE memoryview memoryview::from_buffer(
    void *ptr, ssize_t itemsize, const char* format,
    detail::any_container<ssize_t> shape,
    detail::any_container<ssize_t> strides, bool readonly) {
    size_t ndim = shape->size();
    if (ndim != strides->size())
        pybind11_fail("memoryview: shape length doesn't match strides length");
    ssize_t size = ndim ? 1 : 0;
    for (size_t i = 0; i < ndim; ++i)
        size *= (*shape)[i];
    Py_buffer view;
    view.buf = ptr;
    view.obj = nullptr;
    view.len = size * itemsize;
    view.readonly = static_cast<int>(readonly);
    view.itemsize = itemsize;
    view.format = const_cast<char*>(format);
    view.ndim = static_cast<int>(ndim);
    view.shape = shape->data();
    view.strides = strides->data();
    view.suboffsets = nullptr;
    view.internal = nullptr;
    PyObject* obj = PyMemoryView_FromBuffer(&view);
    if (!obj)
        throw error_already_set();
    return memoryview(object(obj, stolen_t{}));
}
#endif  // DOXYGEN_SHOULD_SKIP_THIS

PYBIND11_INLINE size_t len(handle h) {
    ssize_t result = PyObject_Length(h.ptr());
    if (result < 0)
        pybind11_fail("Unable to compute length of object");
    return (size_t) result;
}

PYBIND11_INLINE size_t len_hint(handle h) {
#if PY_VERSION_HEX >= 0x03040000
    ssize_t result = PyObject_LengthHint(h.ptr(), 0);
#else
    ssize_t result = PyObject_Length(h.ptr());
#endif
    if (result < 0) {
        // Sometimes a length can't be determined at all (eg generators)
        // In which case simply return 0
        PyErr_Clear();
        return 0;
    }
    return (size_t) result;
}

PYBIND11_INLINE str repr(handle h) {
    PyObject *str_value = PyObject_Repr(h.ptr());
    if (!str_value) throw error_already_set();
#if PY_MAJOR_VERSION < 3
    PyObject *unicode = PyUnicode_FromEncodedObject(str_value, "utf-8", nullptr);
    Py_XDECREF(str_value); str_value = unicode;
    if (!str_value) throw error_already_set();
#endif
    return reinterpret_steal<str>(str_value);
}

PYBIND11_INLINE iterator iter(handle obj) {
    PyObject *result = PyObject_GetIter(obj.ptr());
    if (!result) { throw error_already_set(); }
    return reinterpret_steal<iterator>(result);
}

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
