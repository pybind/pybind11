/*
    pybind11/numpy/csr_matrix.h: Basic NumPy support for csr_matrix

    Copyright (c) 2020 Philip Deegan <philip.deegan@gmail.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11/numpy.h"

#include "numpy/arrayobject.h"

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)


#ifndef INDICE_TYPE
#ifdef USE_SPARSE_INDICES_UINT64
#define INDICE_TYPE size_t
#else
#define INDICE_TYPE std::uint32_t
#endif
#endif

template <class T>
class SparseView {
 public:
  SparseView(const T *data, const size_t _size, const INDICE_TYPE *indices)
      : v_data(data), _size(_size), indices(indices) {}
  SparseView(SparseView &&that) : _size(that._size) {
    this->v_data = that.v_data;
    this->indices = that.indices;
  };

 private:
  T const *v_data;
  size_t const _size;
  INDICE_TYPE const *indices;
};

template <class T>
class Sparse2DView {
 public:
  Sparse2DView(T *_data, size_t *_info, INDICE_TYPE *_indices, INDICE_TYPE *_row_indices)
      : v_data(_data), v_indices(_indices), v_row_indices(_row_indices) {
    set_info(_info);
  }
  Sparse2DView(Sparse2DView &&that)
      : v_data(that.v_data), v_indices(that.v_indices), v_row_indices(that.v_row_indices) {
    set_info(that.m_info);
  }

  SparseView<T> row(size_t i) const {
    return SparseView<T>(v_data + v_row_indices[i], v_row_indices[i + 1] - v_row_indices[i],
                         v_indices + v_row_indices[i]);
  }

  SparseView<T> operator[](size_t i) const { return row(i); }

  const size_t &cols() const { return m_info[0]; }
  const size_t &rows() const { return m_info[1]; }
  const size_t &size() const { return m_info[2]; }

 private:
  T *v_data = nullptr;
  size_t m_info[3] = {0, 0, 0};
  INDICE_TYPE *v_indices = nullptr, *v_row_indices = nullptr;

  void set_info(const size_t *_info) {
    m_info[0] = _info[0];
    m_info[1] = _info[1];
    m_info[2] = _info[2];
  }
};

template <typename T>
struct Sparse2dState {
  T *p_data = nullptr;
  INDICE_TYPE *p_indices = nullptr, *p_row_indices = nullptr;
  size_t info[4];
};

template <typename T>
PyObject *sparse2d_to_csr(Sparse2dState<T> &state) {
  auto data_type = NPY_UINT64;
  if (std::is_same<T, double>::value)
    data_type = NPY_DOUBLE;
  else if (std::is_same<T, float>::value)
    data_type = NPY_FLOAT;
  else
    throw std::runtime_error("Unhandeled data type for csr_matrix");

#ifdef USE_SPARSE_INDICES_UINT64
  auto indice_type = NPY_UINT64;
#else
  auto indice_type = NPY_UINT32;
#endif
  auto info = state.info;
  size_t cols = info[0], rows = info[1], size_sparse = info[2];
  npy_intp dims[1];
  dims[0] = size_sparse;
  npy_intp rowDim[1];
  rowDim[0] = rows + 1;

  PyArrayObject *array =
      (PyArrayObject *)PyArray_SimpleNewFromData(1, dims, data_type, state.p_data);
  if (!PyArray_Check(array)) throw std::runtime_error("Array check failed");

  PyArrayObject *indices =
      (PyArrayObject *)PyArray_SimpleNewFromData(1, dims, indice_type, state.p_indices);
  if (!PyArray_Check(indices)) throw std::runtime_error("indices check failed");

  PyArrayObject *row_indices =
      (PyArrayObject *)PyArray_SimpleNewFromData(1, rowDim, indice_type, state.p_row_indices);
  if (!PyArray_Check(row_indices)) throw std::runtime_error("row_indices check failed");

  if (!array) throw std::runtime_error("Array failed");
  if (!indices) throw std::runtime_error("indices failed");
  if (!row_indices) throw std::runtime_error("row_indices failed");

  PyObject *tuple = PyTuple_New(3);
  if (!tuple) throw std::runtime_error("tuple new failed");
  if (!PyTuple_Check(tuple)) throw std::runtime_error("tuple type 1 failed");

  if (PyTuple_SetItem(tuple, 0, (PyObject *)array))
    throw std::runtime_error("tuple PyTuple_SetItem 0 failed");
  if (PyTuple_SetItem(tuple, 1, (PyObject *)indices))
    throw std::runtime_error("tuple PyTuple_SetItem 1 failed");
  if (PyTuple_SetItem(tuple, 2, (PyObject *)row_indices))
    throw std::runtime_error("tuple PyTuple_SetItem 2 failed");
  if (!PyTuple_Check(tuple)) throw std::runtime_error("tuple type 2 failed");

  PyObject *Otuple = PyTuple_New(1);
  if (!Otuple) throw std::runtime_error("Otuple new failed");
  if (PyTuple_SetItem(Otuple, 0, (PyObject *)tuple))
    throw std::runtime_error("Otuple PyTuple_SetItem 0 failed");
  if (!PyTuple_Check(tuple)) throw std::runtime_error("Otuple check failed");

  PyObject *shape = Py_BuildValue("ii", rows, cols);
  if (!shape) throw std::runtime_error("Shape tuple new failed");
  if (!PyTuple_Check(shape)) throw std::runtime_error("shape tuple check failed");

  PyObject *dic = PyDict_New();
  if (!dic) throw std::runtime_error("dict new failed");
  if (PyDict_SetItemString(dic, "shape", shape) == -1)
    throw std::runtime_error("shape set failed on dic");
  if (!PyDict_Check(dic)) throw std::runtime_error("dic is no dic");

  PyObject *scipy_sparse_csr = PyImport_ImportModule("scipy.sparse.csr");
  if (!scipy_sparse_csr) throw std::runtime_error("scipy_sparse_csr failed");
  PyObject *csr_matrix = PyObject_GetAttrString(scipy_sparse_csr, "csr_matrix");
  if (!csr_matrix) throw std::runtime_error("csr_matrix failed");
  if (!PyCallable_Check(csr_matrix)) throw std::runtime_error("csr_matrix check failed");
  PyObject *matrix = PyObject_Call(csr_matrix, Otuple, dic);
  if (!matrix) throw std::runtime_error("matrix failed to call object");

  // csr_matrix destruction does not remove these unless set like this ¯\_(ツ)_/¯
  if (PyObject_SetAttrString(matrix, "_indices", (PyObject *)indices))
    throw std::runtime_error("set indices failed");
  if (PyObject_SetAttrString(matrix, "_row_indices", (PyObject *)row_indices))
    throw std::runtime_error("set row_indices failed");

#if (NPY_API_VERSION >= 7)
  PyArray_ENABLEFLAGS(array, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS(indices, NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS(row_indices, NPY_ARRAY_OWNDATA);
#else
  PyArray_FLAGS(array) |= NPY_OWNDATA;
  PyArray_FLAGS(indices) |= NPY_OWNDATA;
  PyArray_FLAGS(row_indices) |= NPY_OWNDATA;
#endif

  Py_DECREF(array);
  Py_DECREF(indices);
  Py_DECREF(row_indices);

  return matrix;
}

struct PyCSRDescr_Proxy {
  PyObject_HEAD PyObject *typeobj;
  char kind;
  char type;
  char byteorder;
  char flags;
  int type_num;
  int elsize;
  int alignment;
  char *subarray;
  PyObject *fields;
  PyObject *names;
};

struct PyCSR_Proxy {
  PyObject_HEAD char *data;
  int nd = 1;
  ssize_t *dimensions;
  ssize_t *strides;
  PyObject *base;
  PyObject *descr;
  int flags;
};

NAMESPACE_BEGIN(detail)

inline PyCSR_Proxy *csr_proxy(void *ptr) { return reinterpret_cast<PyCSR_Proxy *>(ptr); }
inline const PyCSR_Proxy *csr_proxy(const void *ptr) {
  return reinterpret_cast<const PyCSR_Proxy *>(ptr);
}

inline PyCSRDescr_Proxy *csr_descriptor_proxy(PyObject *ptr) {
  return reinterpret_cast<PyCSRDescr_Proxy *>(ptr);
}

inline const PyCSRDescr_Proxy *csr_descriptor_proxy(const PyObject *ptr) {
  return reinterpret_cast<const PyCSRDescr_Proxy *>(ptr);
}

template <typename T>
struct csr_info_scalar {
  typedef T type;
  static constexpr bool is_csr = false;
  static constexpr bool is_empty = false;
  static constexpr auto extents = _("");
  static void append_extents(list & /* shape */) {}
};
// Computes underlying type and a comma-separated list of extents for array
// types (any mix of std::array and built-in arrays). An array of char is
// treated as scalar because it gets special handling.
template <typename T>
struct csr_info : csr_info_scalar<T> {};

NAMESPACE_END(detail)

class csr : public buffer {
 public:
  PYBIND11_OBJECT_CVT(csr, buffer, detail::npy_api::get().PyArray_Check_, raw_csr)

  enum {
    c_style = detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_,
    f_style = detail::npy_api::NPY_ARRAY_F_CONTIGUOUS_,
    forcecast = detail::npy_api::NPY_ARRAY_FORCECAST_
  };

  csr() : csr({{0}}, static_cast<const double *>(nullptr)) {}

  using ShapeContainer = detail::any_container<ssize_t>;
  using StridesContainer = detail::any_container<ssize_t>;

  // Constructs an csr taking shape/strides from arbitrary container types
  csr(const pybind11::dtype &dt, ShapeContainer shape, StridesContainer strides,
      const void *ptr = nullptr, handle base = handle()) {
    if (strides->empty()) *strides = c_strides(*shape, dt.itemsize());

    auto ndim = shape->size();
    if (ndim != strides->size()) pybind11_fail("NumPy: shape ndim doesn't match strides ndim");
    auto descr = dt;

    int flags = 0;
    if (base && ptr) {
      if (isinstance<csr>(base)) /* Copy flags from base (except ownership bit) */
        flags = reinterpret_borrow<csr>(base).flags() & ~detail::npy_api::NPY_ARRAY_OWNDATA_;
      else
        /* Writable by default, easy to downgrade later on if needed */
        flags = detail::npy_api::NPY_ARRAY_WRITEABLE_;
    }

    auto &api = detail::npy_api::get();
    auto tmp = reinterpret_steal<object>(api.PyArray_NewFromDescr_(
        api.PyArray_Type_, descr.release().ptr(), (int)ndim, shape->data(), strides->data(),
        const_cast<void *>(ptr), flags, nullptr));
    if (!tmp) throw error_already_set();
    if (ptr) {
      if (base) {
        api.PyArray_SetBaseObject_(tmp.ptr(), base.inc_ref().ptr());
      } else {
        tmp = reinterpret_steal<object>(api.PyArray_NewCopy_(tmp.ptr(), -1 /* any order */));
      }
    }
    m_ptr = tmp.release().ptr();
  }

  csr(const pybind11::dtype &dt, ShapeContainer shape, const void *ptr = nullptr,
      handle base = handle())
      : csr(dt, std::move(shape), {}, ptr, base) {}

  template <typename T, typename = detail::enable_if_t<std::is_integral<T>::value &&
                                                       !std::is_same<bool, T>::value>>
  csr(const pybind11::dtype &dt, T count, const void *ptr = nullptr, handle base = handle())
      : csr(dt, {{count}}, ptr, base) {}

  template <typename T>
  csr(ShapeContainer shape, StridesContainer strides, const T *ptr, handle base = handle())
      : csr(pybind11::dtype::of<T>(), std::move(shape), std::move(strides), ptr, base) {}

  template <typename T>
  csr(ShapeContainer shape, const T *ptr, handle base = handle())
      : csr(std::move(shape), {}, ptr, base) {}

  template <typename T>
  explicit csr(ssize_t count, const T *ptr, handle base = handle()) : csr({count}, {}, ptr, base) {}

  explicit csr(const buffer_info &info)
      : csr(pybind11::dtype(info), info.shape, info.strides, info.ptr) {}

  /// Array descriptor (dtype)
  pybind11::dtype dtype() const {
    return reinterpret_borrow<pybind11::dtype>(detail::csr_proxy(m_ptr)->descr);
  }

  /// Total number of elements
  ssize_t size() const {
    return std::accumulate(shape(), shape() + ndim(), (ssize_t)1, std::multiplies<ssize_t>());
  }

  /// Byte size of a single element
  ssize_t itemsize() const {
    return detail::csr_descriptor_proxy(detail::csr_proxy(m_ptr)->descr)->elsize;
  }

  /// Total number of bytes
  ssize_t nbytes() const { return size() * itemsize(); }

  /// Number of dimensions
  ssize_t ndim() const { return detail::csr_proxy(m_ptr)->nd; }

  /// Base object
  object base() const { return reinterpret_borrow<object>(detail::csr_proxy(m_ptr)->base); }

  /// Dimensions of the csr
  const ssize_t *shape() const { return detail::csr_proxy(m_ptr)->dimensions; }

  /// Dimension along a given axis
  ssize_t shape(ssize_t dim) const {
    if (dim >= ndim()) fail_dim_check(dim, "invalid axis");
    return shape()[dim];
  }

  /// Strides of the csr
  const ssize_t *strides() const { return detail::csr_proxy(m_ptr)->strides; }

  /// Stride along a given axis
  ssize_t strides(ssize_t dim) const {
    if (dim >= ndim()) fail_dim_check(dim, "invalid axis");
    return strides()[dim];
  }

  /// Return the NumPy csr flags
  int flags() const { return detail::csr_proxy(m_ptr)->flags; }

  /// If set, the csr is writeable (otherwise the buffer is read-only)
  bool writeable() const {
    return detail::check_flags(m_ptr, detail::npy_api::NPY_ARRAY_WRITEABLE_);
  }

  /// If set, the csr owns the data (will be freed when the csr is deleted)
  bool owndata() const { return detail::check_flags(m_ptr, detail::npy_api::NPY_ARRAY_OWNDATA_); }

  /// Pointer to the contained data. If index is not provided, points to the
  /// beginning of the buffer. May throw if the index would lead to out of bounds access.
  template <typename... Ix>
  const void *data(Ix... index) const {
    return static_cast<const void *>(detail::csr_proxy(m_ptr)->data + offset_at(index...));
  }

  /// Mutable pointer to the contained data. If index is not provided, points to the
  /// beginning of the buffer. May throw if the index would lead to out of bounds access.
  /// May throw if the csr is not writeable.
  template <typename... Ix>
  void *mutable_data(Ix... index) {
    check_writeable();
    return static_cast<void *>(detail::csr_proxy(m_ptr)->data + offset_at(index...));
  }

  /// Byte offset from beginning of the csr to a given index (full or partial).
  /// May throw if the index would lead to out of bounds access.
  template <typename... Ix>
  ssize_t offset_at(Ix... index) const {
    if ((ssize_t)sizeof...(index) > ndim())
      fail_dim_check(sizeof...(index), "too many indices for an csr");
    return byte_offset(ssize_t(index)...);
  }

  ssize_t offset_at() const { return 0; }

  /// Item count from beginning of the csr to a given index (full or partial).
  /// May throw if the index would lead to out of bounds access.
  template <typename... Ix>
  ssize_t index_at(Ix... index) const {
    return offset_at(index...) / itemsize();
  }

  /**
   * Returns a proxy object that provides access to the csr's data without bounds or
   * dimensionality checking.  Will throw if the csr is missing the `writeable` flag.  Use with
   * care: the csr must not be destroyed or reshaped for the duration of the returned object,
   * and the caller must take care not to access invalid dimensions or dimension indices.
   */
  template <typename T, ssize_t Dims = -1>
  detail::unchecked_mutable_reference<T, Dims> mutable_unchecked() & {
    if (Dims >= 0 && ndim() != Dims)
      throw std::domain_error("csr has incorrect number of dimensions: " + std::to_string(ndim()) +
                              "; expected " + std::to_string(Dims));
    return detail::unchecked_mutable_reference<T, Dims>(mutable_data(), shape(), strides(), ndim());
  }

  /**
   * Returns a proxy object that provides const access to the csr's data without bounds or
   * dimensionality checking.  Unlike `mutable_unchecked()`, this does not require that the
   * underlying csr have the `writable` flag.  Use with care: the csr must not be destroyed or
   * reshaped for the duration of the returned object, and the caller must take care not to access
   * invalid dimensions or dimension indices.
   */
  template <typename T, ssize_t Dims = -1>
  detail::unchecked_reference<T, Dims> unchecked() const & {
    if (Dims >= 0 && ndim() != Dims)
      throw std::domain_error("csr has incorrect number of dimensions: " + std::to_string(ndim()) +
                              "; expected " + std::to_string(Dims));
    return detail::unchecked_reference<T, Dims>(data(), shape(), strides(), ndim());
  }

  /// Return a new view with all of the dimensions of length 1 removed
  csr squeeze() {
    auto &api = detail::npy_api::get();
    return reinterpret_steal<csr>(api.PyArray_Squeeze_(m_ptr));
  }

  /// Resize csr to given shape
  /// If refcheck is true and more that one reference exist to this csr
  /// then resize will succeed only if it makes a reshape, i.e. original size doesn't change
  void resize(ShapeContainer new_shape, bool refcheck = true) {
    detail::npy_api::PyArray_Dims d = {new_shape->data(), int(new_shape->size())};
    // try to resize, set ordering param to -1 cause it's not used anyway
    object new_csr = reinterpret_steal<object>(
        detail::npy_api::get().PyArray_Resize_(m_ptr, &d, int(refcheck), -1));
    if (!new_csr) throw error_already_set();
    if (isinstance<csr>(new_csr)) {
      *this = std::move(new_csr);
    }
  }

  /// Ensure that the argument is a NumPy csr
  /// In case of an error, nullptr is returned and the Python error is cleared.
  static csr ensure(handle h, int ExtraFlags = 0) {
    auto result = reinterpret_steal<csr>(raw_csr(h.ptr(), ExtraFlags));
    if (!result) PyErr_Clear();
    return result;
  }

 protected:
  template <typename, typename>
  friend struct detail::npy_format_descriptor;

  void fail_dim_check(ssize_t dim, const std::string &msg) const {
    throw index_error(msg + ": " + std::to_string(dim) + " (ndim = " + std::to_string(ndim()) +
                      ")");
  }

  template <typename... Ix>
  ssize_t byte_offset(Ix... index) const {
    check_dimensions(index...);
    return detail::byte_offset_unsafe(strides(), ssize_t(index)...);
  }

  void check_writeable() const {
    if (!writeable()) throw std::domain_error("csr is not writeable");
  }

  // Default, C-style strides
  static std::vector<ssize_t> c_strides(const std::vector<ssize_t> &shape, ssize_t itemsize) {
    auto ndim = shape.size();
    std::vector<ssize_t> strides(ndim, itemsize);
    if (ndim > 0)
      for (size_t i = ndim - 1; i > 0; --i) strides[i - 1] = strides[i] * shape[i];
    return strides;
  }

  // F-style strides; default when constructing an csr_t with `ExtraFlags & f_style`
  static std::vector<ssize_t> f_strides(const std::vector<ssize_t> &shape, ssize_t itemsize) {
    auto ndim = shape.size();
    std::vector<ssize_t> strides(ndim, itemsize);
    for (size_t i = 1; i < ndim; ++i) strides[i] = strides[i - 1] * shape[i - 1];
    return strides;
  }

  template <typename... Ix>
  void check_dimensions(Ix... index) const {
    check_dimensions_impl(ssize_t(0), shape(), ssize_t(index)...);
  }

  void check_dimensions_impl(ssize_t, const ssize_t *) const {}

  template <typename... Ix>
  void check_dimensions_impl(ssize_t axis, const ssize_t *shape, ssize_t i, Ix... index) const {
    if (i >= *shape) {
      throw index_error(std::string("index ") + std::to_string(i) + " is out of bounds for axis " +
                        std::to_string(axis) + " with size " + std::to_string(*shape));
    }
    check_dimensions_impl(axis + 1, shape + 1, index...);
  }

  /// Create csr from any object -- always returns a new reference
  static PyObject *raw_csr(PyObject *ptr, int ExtraFlags = 0) {
    if (ptr == nullptr) {
      PyErr_SetString(PyExc_ValueError, "cannot create a pybind11::csr from a nullptr");
      return nullptr;
    }
    return detail::npy_api::get().PyArray_FromAny_(
        ptr, nullptr, 0, 0, detail::npy_api::NPY_ARRAY_ENSUREARRAY_ | ExtraFlags, nullptr);
  }
};

template <typename T, int ExtraFlags = csr::forcecast>
class csr_t : public csr {
 private:
  struct private_ctor {};
  // Delegating constructor needed when both moving and accessing in the same constructor
  csr_t(private_ctor, ShapeContainer &&shape, StridesContainer &&strides, const T *ptr, handle base)
      : csr(std::move(shape), std::move(strides), ptr, base) {}

 public:
  static_assert(!detail::csr_info<T>::is_csr, "Array types cannot be used with csr_t");

  using value_type = T;

  std::vector<size_t> _info;
  std::shared_ptr<Sparse2DView<T>> m_data_ptr;
  std::shared_ptr<Sparse2dState<T>> mSparse2dState;

  csr_t() : csr(0, static_cast<const T *>(nullptr)) {}
  csr_t(handle h, borrowed_t) : csr(h, borrowed_t{}), _info(3) {
    auto *obj = h.ptr();
    auto *obj_shape = PyObject_GetAttrString(obj, "shape");
    auto *obj_indptr = (PyArrayObject *)PyObject_GetAttrString(obj, "indptr");
    auto *obj_indices = (PyArrayObject *)PyObject_GetAttrString(obj, "indices");
    auto *obj_data = (PyArrayObject *)PyObject_GetAttrString(obj, "data");

    if (obj_shape == NULL || obj_indptr == NULL || obj_indices == NULL || obj_data == NULL) {
      PyErr_SetString(PyExc_ValueError,
                      "Expecting a 2d sparse numpy array (i.e., a python object with 3 fields "
                      "'indptr', 'indices' and 'data')");
      if (obj_shape) Py_DECREF(obj_shape);
      if (obj_indptr) Py_DECREF(obj_indptr);
      if (obj_indices) Py_DECREF(obj_indices);
      if (obj_data) Py_DECREF(obj_data);
      throw std::runtime_error("ERROR 1");
    }

    if (!PyArray_IS_C_CONTIGUOUS(obj_data) || !PyArray_IS_C_CONTIGUOUS(obj_indptr) ||
        !PyArray_IS_C_CONTIGUOUS(obj_indices)) {
      PyErr_SetString(
          PyExc_ValueError,
          "The fields indptr, indices and data of sparse matrix must be contiguous numpy arrays.");
      Py_DECREF(obj_indptr);
      Py_DECREF(obj_indices);
      Py_DECREF(obj_data);
      Py_DECREF(obj_shape);
      throw std::runtime_error("ERROR 2");
    }

    PyObject *obj_nrows = PyTuple_GET_ITEM(obj_shape, 0);
    PyObject *obj_ncols = PyTuple_GET_ITEM(obj_shape, 1);
    _info[0] = PyLong_AsLong(obj_ncols);
    _info[1] = PyLong_AsLong(obj_nrows);
    _info[2] = PyArray_DIM(obj_data, 0);

    auto *data = (T *)PyArray_DATA(obj_data);
    auto *indices = (INDICE_TYPE *)PyArray_DATA(obj_indices);
    auto *row_indices = (INDICE_TYPE *)PyArray_DATA(obj_indptr);

    m_data_ptr = std::make_shared<Sparse2DView<T>>(data, _info.data(), indices, row_indices);
  }
  csr_t(handle h, stolen_t) : csr(h, stolen_t{}) {}

  PYBIND11_DEPRECATED("Use csr_t<T>::ensure() instead")
  csr_t(handle h, bool is_borrowed) : csr(raw_csr_t(h.ptr()), stolen_t{}) {
    if (!m_ptr) PyErr_Clear();
    if (!is_borrowed) Py_XDECREF(h.ptr());
  }

  csr_t(const object &o) : csr(raw_csr_t(o.ptr()), stolen_t{}) {
    if (!m_ptr) throw error_already_set();
  }

  explicit csr_t(const buffer_info &info) : csr(info) {}

  explicit csr_t(std::shared_ptr<Sparse2dState<T>> &&state)
      : csr(reinterpret_steal<object>(sparse2d_to_csr<T>(*state.get()))), mSparse2dState(state) {
    m_data_ptr =
        std::make_shared<Sparse2DView<T>>(mSparse2dState->p_data, mSparse2dState->info.data(),
                                          mSparse2dState->p_indices, mSparse2dState->p_row_indices);
  }

  csr_t(ShapeContainer shape, StridesContainer strides, const T *ptr = nullptr,
        handle base = handle())
      : csr(std::move(shape), std::move(strides), ptr, base) {}

  explicit csr_t(ShapeContainer shape, const T *ptr = nullptr, handle base = handle())
      : csr_t(private_ctor{}, std::move(shape),
              ExtraFlags & f_style ? f_strides(*shape, itemsize()) : c_strides(*shape, itemsize()),
              ptr, base) {}

  explicit csr_t(size_t count, const T *ptr = nullptr, handle base = handle())
      : csr({count}, {}, ptr, base) {}

  constexpr ssize_t itemsize() const { return sizeof(T); }

  template <typename... Ix>
  ssize_t index_at(Ix... index) const {
    return offset_at(index...) / itemsize();
  }

  template <typename... Ix>
  const T *data(Ix... index) const {
    return static_cast<const T *>(csr::data(index...));
  }

  template <typename... Ix>
  T *mutable_data(Ix... index) {
    return static_cast<T *>(csr::mutable_data(index...));
  }

  // Reference to element at a given index
  template <typename... Ix>
  const T &at(Ix... index) const {
    if (sizeof...(index) != ndim()) fail_dim_check(sizeof...(index), "index dimension mismatch");
    return *(static_cast<const T *>(csr::data()) + byte_offset(ssize_t(index)...) / itemsize());
  }

  // Mutable reference to element at a given index
  template <typename... Ix>
  T &mutable_at(Ix... index) {
    if (sizeof...(index) != ndim()) fail_dim_check(sizeof...(index), "index dimension mismatch");
    return *(static_cast<T *>(csr::mutable_data()) + byte_offset(ssize_t(index)...) / itemsize());
  }

  /**
   * Returns a proxy object that provides access to the csr's data without bounds or
   * dimensionality checking.  Will throw if the csr is missing the `writeable` flag.  Use with
   * care: the csr must not be destroyed or reshaped for the duration of the returned object,
   * and the caller must take care not to access invalid dimensions or dimension indices.
   */
  template <ssize_t Dims = -1>
  detail::unchecked_mutable_reference<T, Dims> mutable_unchecked() & {
    return csr::mutable_unchecked<T, Dims>();
  }

  /**
   * Returns a proxy object that provides const access to the csr's data without bounds or
   * dimensionality checking.  Unlike `unchecked()`, this does not require that the underlying
   * csr have the `writable` flag.  Use with care: the csr must not be destroyed or reshaped
   * for the duration of the returned object, and the caller must take care not to access invalid
   * dimensions or dimension indices.
   */
  template <ssize_t Dims = -1>
  detail::unchecked_reference<T, Dims> unchecked() const & {
    return csr::unchecked<T, Dims>();
  }

  /// Ensure that the argument is a NumPy csr of the correct dtype (and if not, try to convert
  /// it).  In case of an error, nullptr is returned and the Python error is cleared.
  static csr_t ensure(handle h) {
    auto result = reinterpret_steal<csr_t>(raw_csr_t(h.ptr()));
    if (!result) PyErr_Clear();
    return result;
  }

  // TODO FIX
  static bool check_(handle /*h*/) {
    // const auto &api = detail::npy_api::get();
    return 1;  // api.PyArray_Check_(h.ptr()) &&
               // api.PyArray_EquivTypes_(detail::csr_proxy(h.ptr())->descr, dtype::of<T>().ptr());
  }

 protected:
  /// Create csr from any object -- always returns a new reference
  static PyObject *raw_csr_t(PyObject *ptr) {
    if (ptr == nullptr) {
      PyErr_SetString(PyExc_ValueError, "cannot create a pybind11::csr_t from a nullptr");
      return nullptr;
    }

    return detail::npy_api::get().PyArray_FromAny_(
        ptr, dtype::of<T>().release().ptr(), 0, 0,
        detail::npy_api::NPY_ARRAY_ENSUREARRAY_ | ExtraFlags, nullptr);
  }
};



NAMESPACE_END(PYBIND11_NAMESPACE)
