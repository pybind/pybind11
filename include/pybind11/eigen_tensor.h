/*
    eigen_tensor.h: Transparent conversion for Eigen tensors

    Adaptiation of the eigen vector/matrix wrapper by:
    Wenzel Jakob <wenzel.jakob@epfl.ch>

    Copyright (c) 2016 Hugo Strand <hugo.strand@gmail.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "numpy.h"

#if defined(__GNUG__) || defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

#include <unsupported/Eigen/CXX11/Tensor>

#if defined(__GNUG__) || defined(__clang__)
#  pragma GCC diagnostic pop
#endif

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

template <typename T> class is_eigen_tensor {
private:

  // Can not get Eige::TensorBase working
  // here is a workaround for Eigen::Tenor and Eigen::TensorMap
  
  template<typename Scalar, int Rank, int Options>
  static std::true_type test(Eigen::Tensor<Scalar, Rank, Options>);

  template<typename Scalar, int Rank, int Options>
  static std::true_type test(Eigen::TensorMap<Eigen::Tensor<Scalar, Rank, Options> >);
  
  static std::false_type test(...);
public:
  static constexpr bool value = decltype(test(std::declval<T>()))::value;
};

template<typename Type>
struct type_caster<Type,
  typename std::enable_if<is_eigen_tensor<Type>::value>::type> {

  typedef typename Type::Scalar Scalar;
  static constexpr int Rank = Type::NumIndices;
    
  bool load(handle src, bool) {

    array_t<Scalar> buffer(src, true);
    if (!buffer.check()) return false;

    buffer_info info = buffer.request();

    if (info.ndim != Rank) return false;

    // -- Check that the strides agrees with the tensor index order
    if( Type::Options == Eigen::RowMajor ) {
	
      // the strides must be in decreasing order
      for(int idx = 0; idx < info.strides.size()-1; idx++)
	if(info.strides[idx] < info.strides[idx+1])
	  return false; // buffer is not RowMajor
	
    } else if ( Type::Options == Eigen::ColMajor ) {

      // the strides must be in increasing order
      for(int idx = 0; idx < info.strides.size()-1; idx++)
	if(info.strides[idx] > info.strides[idx+1])
	  return false; // buffer is not ColMajor

    } else {
      return false;
    }
      
    Eigen::array<Eigen::Index, Rank> shape;
    std::copy_n(info.shape.begin(), Rank, shape.begin());
    value = Eigen::TensorMap<Type>((Scalar *) info.ptr, shape);
    
    return true;
  }

  static handle cast(const Type *src,
		     return_value_policy policy,
		     handle parent) {
    return cast(*src, policy, parent);
  }

  static handle cast(const Type &src,
		     return_value_policy /* policy */,
		     handle /* parent */) {

    size_t N = src.NumDimensions;
    std::vector<size_t> strides(N);
    std::vector<size_t> dimensions(N);

    for(int idx = 0; idx < dimensions.size(); idx++)
      dimensions[idx] = src.dimension(idx);

    size_t stride = 1;
    if( Type::Options == Eigen::RowMajor ) {
      for(int idx = 0; idx < strides.size(); idx++) {
	strides[N-idx-1] = stride * sizeof(Scalar);
	stride *= dimensions[N-idx-1];
      }
    } else { // -- COLUMN major assumption (Default)
      for(int idx = 0; idx < strides.size(); idx++) {
	strides[idx] = stride * sizeof(Scalar);
	stride *= dimensions[idx];
      }
    }

    return array(buffer_info(
      (void *) src.data(), sizeof(Scalar),
      format_descriptor<Scalar>::value,
      src.NumDimensions, dimensions, strides)).release();    
  }

    template <typename _T> using cast_op_type
      = pybind11::detail::cast_op_type<_T>;

    static PYBIND11_DESCR name() {
        return _("numpy.ndarray[dtype=")
	  + npy_format_descriptor<Scalar>::name() +
	  _(", shape=(...)]");
    }

    operator Type*() { return &value; }
    operator Type&() { return value; }

protected:
    Type value;
};

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
