#include "numpy.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_INLINE numpy_type_info *numpy_internals::get_type_info(const std::type_info& tinfo, bool throw_if_missing) {
    auto it = registered_dtypes.find(std::type_index(tinfo));
    if (it != registered_dtypes.end())
        return &(it->second);
    if (throw_if_missing)
        pybind11_fail(std::string("NumPy type info missing for ") + tinfo.name());
    return nullptr;
}

PYBIND11_INLINE void load_numpy_internals(numpy_internals* &ptr) {
    ptr = &get_or_create_shared_data<numpy_internals>("_numpy_internals");
}

PYBIND11_INLINE numpy_internals& get_numpy_internals() {
    static numpy_internals* ptr = nullptr;
    if (!ptr)
        load_numpy_internals(ptr);
    return *ptr;
}

PYBIND11_INLINE npy_api& npy_api::get() {
    static npy_api api = lookup();
    return api;
}

PYBIND11_INLINE bool npy_api::PyArray_Check_(PyObject *obj) const {
    return (bool) PyObject_TypeCheck(obj, PyArray_Type_);
}

PYBIND11_INLINE bool npy_api::PyArrayDescr_Check_(PyObject *obj) const {
    return (bool) PyObject_TypeCheck(obj, PyArrayDescr_Type_);
}

PYBIND11_INLINE npy_api npy_api::lookup() {
    module m = module::import("numpy.core.multiarray");
    auto c = m.attr("_ARRAY_API");
#if PY_MAJOR_VERSION >= 3
    void **api_ptr = (void **) PyCapsule_GetPointer(c.ptr(), NULL);
#else
    void **api_ptr = (void **) PyCObject_AsVoidPtr(c.ptr());
#endif
    npy_api api;
#define DECL_NPY_API(Func) api.Func##_ = (decltype(api.Func##_)) api_ptr[API_##Func];
    DECL_NPY_API(PyArray_GetNDArrayCFeatureVersion);
    if (api.PyArray_GetNDArrayCFeatureVersion_() < 0x7)
        pybind11_fail("pybind11 numpy support requires numpy >= 1.7.0");
    DECL_NPY_API(PyArray_Type);
    DECL_NPY_API(PyVoidArrType_Type);
    DECL_NPY_API(PyArrayDescr_Type);
    DECL_NPY_API(PyArray_DescrFromType);
    DECL_NPY_API(PyArray_DescrFromScalar);
    DECL_NPY_API(PyArray_FromAny);
    DECL_NPY_API(PyArray_Resize);
    DECL_NPY_API(PyArray_CopyInto);
    DECL_NPY_API(PyArray_NewCopy);
    DECL_NPY_API(PyArray_NewFromDescr);
    DECL_NPY_API(PyArray_DescrNewFromType);
    DECL_NPY_API(PyArray_DescrConverter);
    DECL_NPY_API(PyArray_EquivTypes);
    DECL_NPY_API(PyArray_GetArrayParamsFromObject);
    DECL_NPY_API(PyArray_Squeeze);
    DECL_NPY_API(PyArray_SetBaseObject);
#undef DECL_NPY_API
    return api;
}

PYBIND11_INLINE PyArray_Proxy* array_proxy(void* ptr) {
    return reinterpret_cast<PyArray_Proxy*>(ptr);
}

PYBIND11_INLINE const PyArray_Proxy* array_proxy(const void* ptr) {
    return reinterpret_cast<const PyArray_Proxy*>(ptr);
}

PYBIND11_INLINE PyArrayDescr_Proxy* array_descriptor_proxy(PyObject* ptr) {
   return reinterpret_cast<PyArrayDescr_Proxy*>(ptr);
}

PYBIND11_INLINE const PyArrayDescr_Proxy* array_descriptor_proxy(const PyObject* ptr) {
   return reinterpret_cast<const PyArrayDescr_Proxy*>(ptr);
}

PYBIND11_INLINE bool check_flags(const void* ptr, int flag) {
    return (flag == (array_proxy(ptr)->flags & flag));
}

PYBIND11_NAMESPACE_END(detail)

PYBIND11_INLINE dtype::dtype(const buffer_info &info) {
    dtype descr(_dtype_from_pep3118()(PYBIND11_STR_TYPE(info.format)));
    // If info.itemsize == 0, use the value calculated from the format string
    m_ptr = descr.strip_padding(info.itemsize ? info.itemsize : descr.itemsize()).release().ptr();
}

PYBIND11_INLINE dtype::dtype(const std::string &format) {
    m_ptr = from_args(pybind11::str(format)).release().ptr();
}

PYBIND11_INLINE dtype::dtype(const char *format) : dtype(std::string(format)) { }

PYBIND11_INLINE dtype::dtype(list names, list formats, list offsets, ssize_t itemsize) {
    dict args;
    args["names"] = names;
    args["formats"] = formats;
    args["offsets"] = offsets;
    args["itemsize"] = pybind11::int_(itemsize);
    m_ptr = from_args(args).release().ptr();
}

PYBIND11_INLINE dtype dtype::from_args(object args) {
    PyObject *ptr = nullptr;
    if (!detail::npy_api::get().PyArray_DescrConverter_(args.ptr(), &ptr) || !ptr)
        throw error_already_set();
    return reinterpret_steal<dtype>(ptr);
}

PYBIND11_INLINE ssize_t dtype::itemsize() const {
    return detail::array_descriptor_proxy(m_ptr)->elsize;
}

PYBIND11_INLINE bool dtype::has_fields() const {
    return detail::array_descriptor_proxy(m_ptr)->names != nullptr;
}

PYBIND11_INLINE char dtype::kind() const {
    return detail::array_descriptor_proxy(m_ptr)->kind;
}

PYBIND11_INLINE object dtype::_dtype_from_pep3118() {
    static PyObject *obj = module::import("numpy.core._internal")
        .attr("_dtype_from_pep3118").cast<object>().release().ptr();
    return reinterpret_borrow<object>(obj);
}

PYBIND11_INLINE dtype dtype::strip_padding(ssize_t itemsize) {
    // Recursively strip all void fields with empty names that are generated for
    // padding fields (as of NumPy v1.11).
    if (!has_fields())
        return *this;

    struct field_descr { PYBIND11_STR_TYPE name; object format; pybind11::int_ offset; };
    std::vector<field_descr> field_descriptors;

    for (auto field : attr("fields").attr("items")()) {
        auto spec = field.cast<tuple>();
        auto name = spec[0].cast<pybind11::str>();
        auto format = spec[1].cast<tuple>()[0].cast<dtype>();
        auto offset = spec[1].cast<tuple>()[1].cast<pybind11::int_>();
        if (!len(name) && format.kind() == 'V')
            continue;
        field_descriptors.push_back({(PYBIND11_STR_TYPE) name, format.strip_padding(format.itemsize()), offset});
    }

    std::sort(field_descriptors.begin(), field_descriptors.end(),
                [](const field_descr& a, const field_descr& b) {
                    return a.offset.cast<int>() < b.offset.cast<int>();
                });

    list names, formats, offsets;
    for (auto& descr : field_descriptors) {
        names.append(descr.name);
        formats.append(descr.format);
        offsets.append(descr.offset);
    }
    return dtype(names, formats, offsets, itemsize);
}

PYBIND11_INLINE array::array(const pybind11::dtype &dt, ShapeContainer shape, StridesContainer strides,
        const void *ptr, handle base) {

    if (strides->empty())
        *strides = c_strides(*shape, dt.itemsize());

    auto ndim = shape->size();
    if (ndim != strides->size())
        pybind11_fail("NumPy: shape ndim doesn't match strides ndim");
    auto descr = dt;

    int flags = 0;
    if (base && ptr) {
        if (isinstance<array>(base))
            /* Copy flags from base (except ownership bit) */
            flags = reinterpret_borrow<array>(base).flags() & ~detail::npy_api::NPY_ARRAY_OWNDATA_;
        else
            /* Writable by default, easy to downgrade later on if needed */
            flags = detail::npy_api::NPY_ARRAY_WRITEABLE_;
    }

    auto &api = detail::npy_api::get();
    auto tmp = reinterpret_steal<object>(api.PyArray_NewFromDescr_(
        api.PyArray_Type_, descr.release().ptr(), (int) ndim, shape->data(), strides->data(),
        const_cast<void *>(ptr), flags, nullptr));
    if (!tmp)
        throw error_already_set();
    if (ptr) {
        if (base) {
            api.PyArray_SetBaseObject_(tmp.ptr(), base.inc_ref().ptr());
        } else {
            tmp = reinterpret_steal<object>(api.PyArray_NewCopy_(tmp.ptr(), -1 /* any order */));
        }
    }
    m_ptr = tmp.release().ptr();
}

PYBIND11_INLINE array::array(const pybind11::dtype &dt, ShapeContainer shape, const void *ptr, handle base)
    : array(dt, std::move(shape), {}, ptr, base) { }

PYBIND11_INLINE array::array(const buffer_info &info, handle base)
: array(pybind11::dtype(info), info.shape, info.strides, info.ptr, base) { }

PYBIND11_INLINE pybind11::dtype array::dtype() const {
    return reinterpret_borrow<pybind11::dtype>(detail::array_proxy(m_ptr)->descr);
}

PYBIND11_INLINE ssize_t array::size() const {
    return std::accumulate(shape(), shape() + ndim(), (ssize_t) 1, std::multiplies<ssize_t>());
}

PYBIND11_INLINE ssize_t array::itemsize() const {
    return detail::array_descriptor_proxy(detail::array_proxy(m_ptr)->descr)->elsize;
}

PYBIND11_INLINE ssize_t array::nbytes() const {
    return size() * itemsize();
}

PYBIND11_INLINE ssize_t array::ndim() const {
    return detail::array_proxy(m_ptr)->nd;
}

PYBIND11_INLINE object array::base() const {
    return reinterpret_borrow<object>(detail::array_proxy(m_ptr)->base);
}

PYBIND11_INLINE const ssize_t* array::shape() const {
    return detail::array_proxy(m_ptr)->dimensions;
}

PYBIND11_INLINE ssize_t array::shape(ssize_t dim) const {
    if (dim >= ndim())
        fail_dim_check(dim, "invalid axis");
    return shape()[dim];
}

PYBIND11_INLINE const ssize_t* array::strides() const {
    return detail::array_proxy(m_ptr)->strides;
}

PYBIND11_INLINE ssize_t array::strides(ssize_t dim) const {
    if (dim >= ndim())
        fail_dim_check(dim, "invalid axis");
    return strides()[dim];
}

PYBIND11_INLINE int array::flags() const {
    return detail::array_proxy(m_ptr)->flags;
}

PYBIND11_INLINE bool array::writeable() const {
    return detail::check_flags(m_ptr, detail::npy_api::NPY_ARRAY_WRITEABLE_);
}

PYBIND11_INLINE bool array::owndata() const {
    return detail::check_flags(m_ptr, detail::npy_api::NPY_ARRAY_OWNDATA_);
}

PYBIND11_INLINE array array::squeeze() {
    auto& api = detail::npy_api::get();
    return reinterpret_steal<array>(api.PyArray_Squeeze_(m_ptr));
}

PYBIND11_INLINE void array::resize(ShapeContainer new_shape, bool refcheck) {
    detail::npy_api::PyArray_Dims d = {
        new_shape->data(), int(new_shape->size())
    };
    // try to resize, set ordering param to -1 cause it's not used anyway
    object new_array = reinterpret_steal<object>(
        detail::npy_api::get().PyArray_Resize_(m_ptr, &d, int(refcheck), -1)
    );
    if (!new_array) throw error_already_set();
    if (isinstance<array>(new_array)) { *this = std::move(new_array); }
}

PYBIND11_INLINE array array::ensure(handle h, int ExtraFlags) {
    auto result = reinterpret_steal<array>(raw_array(h.ptr(), ExtraFlags));
    if (!result)
        PyErr_Clear();
    return result;
}

PYBIND11_INLINE void array::fail_dim_check(ssize_t dim, const std::string& msg) const {
    throw index_error(msg + ": " + std::to_string(dim) +
                        " (ndim = " + std::to_string(ndim()) + ")");
}

PYBIND11_INLINE std::vector<ssize_t> array::c_strides(const std::vector<ssize_t> &shape, ssize_t itemsize) {
    auto ndim = shape.size();
    std::vector<ssize_t> strides(ndim, itemsize);
    if (ndim > 0)
        for (size_t i = ndim - 1; i > 0; --i)
            strides[i - 1] = strides[i] * shape[i];
    return strides;
}

PYBIND11_INLINE std::vector<ssize_t> array::f_strides(const std::vector<ssize_t> &shape, ssize_t itemsize) {
    auto ndim = shape.size();
    std::vector<ssize_t> strides(ndim, itemsize);
    for (size_t i = 1; i < ndim; ++i)
        strides[i] = strides[i - 1] * shape[i - 1];
    return strides;
}

PYBIND11_INLINE PyObject *array::raw_array(PyObject *ptr, int ExtraFlags) {
    if (ptr == nullptr) {
        PyErr_SetString(PyExc_ValueError, "cannot create a pybind11::array from a nullptr");
        return nullptr;
    }
    return detail::npy_api::get().PyArray_FromAny_(
        ptr, nullptr, 0, 0, detail::npy_api::NPY_ARRAY_ENSUREARRAY_ | ExtraFlags, nullptr);
}

PYBIND11_NAMESPACE_BEGIN(detail)


PYBIND11_INLINE void register_structured_dtype(
    any_container<field_descriptor> fields,
    const std::type_info& tinfo, ssize_t itemsize,
    bool (*direct_converter)(PyObject *, void *&)) {

    auto& numpy_internals = get_numpy_internals();
    if (numpy_internals.get_type_info(tinfo, false))
        pybind11_fail("NumPy: dtype is already registered");

    // Use ordered fields because order matters as of NumPy 1.14:
    // https://docs.scipy.org/doc/numpy/release.html#multiple-field-indexing-assignment-of-structured-arrays
    std::vector<field_descriptor> ordered_fields(std::move(fields));
    std::sort(ordered_fields.begin(), ordered_fields.end(),
        [](const field_descriptor &a, const field_descriptor &b) { return a.offset < b.offset; });

    list names, formats, offsets;
    for (auto& field : ordered_fields) {
        if (!field.descr)
            pybind11_fail(std::string("NumPy: unsupported field dtype: `") +
                            field.name + "` @ " + tinfo.name());
        names.append(PYBIND11_STR_TYPE(field.name));
        formats.append(field.descr);
        offsets.append(pybind11::int_(field.offset));
    }
    auto dtype_ptr = pybind11::dtype(names, formats, offsets, itemsize).release().ptr();

    // There is an existing bug in NumPy (as of v1.11): trailing bytes are
    // not encoded explicitly into the format string. This will supposedly
    // get fixed in v1.12; for further details, see these:
    // - https://github.com/numpy/numpy/issues/7797
    // - https://github.com/numpy/numpy/pull/7798
    // Because of this, we won't use numpy's logic to generate buffer format
    // strings and will just do it ourselves.
    ssize_t offset = 0;
    std::ostringstream oss;
    // mark the structure as unaligned with '^', because numpy and C++ don't
    // always agree about alignment (particularly for complex), and we're
    // explicitly listing all our padding. This depends on none of the fields
    // overriding the endianness. Putting the ^ in front of individual fields
    // isn't guaranteed to work due to https://github.com/numpy/numpy/issues/9049
    oss << "^T{";
    for (auto& field : ordered_fields) {
        if (field.offset > offset)
            oss << (field.offset - offset) << 'x';
        oss << field.format << ':' << field.name << ':';
        offset = field.offset + field.size;
    }
    if (itemsize > offset)
        oss << (itemsize - offset) << 'x';
    oss << '}';
    auto format_str = oss.str();

    // Sanity check: verify that NumPy properly parses our buffer format string
    auto& api = npy_api::get();
    auto arr =  array(buffer_info(nullptr, itemsize, format_str, 1));
    if (!api.PyArray_EquivTypes_(dtype_ptr, arr.dtype().ptr()))
        pybind11_fail("NumPy: invalid buffer descriptor!");

    auto tindex = std::type_index(tinfo);
    numpy_internals.registered_dtypes[tindex] = { dtype_ptr, format_str };
    get_internals().direct_conversions[tindex].push_back(direct_converter);
}

PYBIND11_INLINE common_iterator::common_iterator(void* ptr, const container_type& strides, const container_type& shape)
    : p_ptr(reinterpret_cast<char*>(ptr)), m_strides(strides.size()) {
    m_strides.back() = static_cast<value_type>(strides.back());
    for (size_type i = m_strides.size() - 1; i != 0; --i) {
        size_type j = i - 1;
        value_type s = static_cast<value_type>(shape[i]);
        m_strides[j] = strides[j] + m_strides[i] - strides[i] * s;
    }
}

PYBIND11_INLINE void common_iterator::increment(size_type dim) {
    p_ptr += m_strides[dim];
}

PYBIND11_INLINE void* common_iterator::data() const {
    return p_ptr;
}
PYBIND11_NAMESPACE_END(detail)

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
