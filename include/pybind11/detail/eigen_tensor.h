
/*
    pybind11/eigen_tensor.h: Transparent conversion for Eigen tensors

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

/* HINT: To suppress warnings originating from the Eigen headers, use -isystem.
   See also:
       https://stackoverflow.com/questions/2579576/i-dir-vs-isystem-dir
       https://stackoverflow.com/questions/1741816/isystem-for-ms-visual-studio-c-compiler
*/

#include "../numpy.h"

// The C4127 suppression was introduced for Eigen 3.4.0. In theory we could
// make it version specific, or even remove it later, but considering that
// 1. C4127 is generally far more distracting than useful for modern template code, and
// 2. we definitely want to ignore any MSVC warnings originating from Eigen code,
//    it is probably best to keep this around indefinitely.
#if defined(_MSC_VER)
#    pragma warning(push)
#    pragma warning(disable : 4554) // Tensor.h warning
#    pragma warning(disable : 4127) // Tensor.h warning
#elif defined(__MINGW32__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include <unsupported/Eigen/CXX11/Tensor>

#if defined(_MSC_VER)
#    pragma warning(pop)
#elif defined(__MINGW32__)
#    pragma GCC diagnostic pop
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_NAMESPACE_BEGIN(detail)

template <typename T>
constexpr int compute_array_flag_from_tensor() {
    static_assert((static_cast<int>(T::Layout) == static_cast<int>(Eigen::RowMajor))
                      || (static_cast<int>(T::Layout) == static_cast<int>(Eigen::ColMajor)),
                  "Layout must be row or column major");
    return (static_cast<int>(T::Layout) == static_cast<int>(Eigen::RowMajor)) ? array::c_style
                                                                              : array::f_style;
}

template <typename T>
struct eigen_tensor_helper {};

template <typename Scalar_, int NumIndices_, int Options_, typename IndexType>
struct eigen_tensor_helper<Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType>> {
    using Type = Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType>;
    using ValidType = void;

    static Eigen::DSizes<typename Type::Index, Type::NumIndices> get_shape(const Type &f) {
        return f.dimensions();
    }

    static constexpr bool
    is_correct_shape(const Eigen::DSizes<typename Type::Index, Type::NumIndices> & /*shape*/) {
        return true;
    }

    template <typename T>
    struct helper {};

    template <size_t... Is>
    struct helper<index_sequence<Is...>> {
        // Hack to work around gcc 4.8 bugs. Feel free to remove when we drop gcc 4.8 support.
        static constexpr descr<sizeof...(Is) * 3 - 2> value
            = concat(const_name(((void) Is, "?"))...);
    };

    static constexpr auto dimensions_descriptor
        = helper<decltype(make_index_sequence<Type::NumIndices>())>::value;

    template <typename... Args>
    static Type *alloc(Args &&...args) {
        return new Type(std::forward<Args>(args)...);
    }

    static void free(Type *tensor) { delete tensor; }
};

template <typename Scalar_, typename std::ptrdiff_t... Indices, int Options_, typename IndexType>
struct eigen_tensor_helper<
    Eigen::TensorFixedSize<Scalar_, Eigen::Sizes<Indices...>, Options_, IndexType>> {
    using Type = Eigen::TensorFixedSize<Scalar_, Eigen::Sizes<Indices...>, Options_, IndexType>;
    using ValidType = void;

    static constexpr Eigen::DSizes<typename Type::Index, Type::NumIndices> get_shape(const Type & /*f*/) {
        return get_shape();
    }

    static constexpr Eigen::DSizes<typename Type::Index, Type::NumIndices> get_shape() {
        return Eigen::DSizes<typename Type::Index, Type::NumIndices>(Indices...);
    }

    static bool is_correct_shape(const Eigen::DSizes<typename Type::Index, Type::NumIndices> &shape) {
        return get_shape() == shape;
    }

    static constexpr auto dimensions_descriptor = concat(const_name<Indices>()...);

    template <typename... Args>
    static Type *alloc(Args &&...args) {
        Eigen::aligned_allocator<Type> allocator;
        return ::new (allocator.allocate(1)) Type(std::forward<Args>(args)...);
    }

    static void free(Type *tensor) {
        Eigen::aligned_allocator<Type> allocator;
        tensor->~Type();
        allocator.deallocate(tensor, 1);
    }
};

template <typename Type>
struct get_tensor_descriptor {
    static constexpr auto value
        = const_name("numpy.ndarray[") + npy_format_descriptor<typename Type::Scalar>::name
          + const_name("[") + eigen_tensor_helper<Type>::dimensions_descriptor
          + const_name("], flags.writeable, ")
          + const_name<(int) Type::Layout == (int) Eigen::RowMajor>("flags.c_contiguous",
                                                                 "flags.f_contiguous");
};

template <typename Type>
struct type_caster<Type, typename eigen_tensor_helper<Type>::ValidType> {
    using Helper = eigen_tensor_helper<Type>;
    PYBIND11_TYPE_CASTER(Type, get_tensor_descriptor<Type>::value);

    bool load(handle src, bool /*convert*/) {
        array_t<typename Type::Scalar, compute_array_flag_from_tensor<Type>()> arr(
            reinterpret_borrow<object>(src));

        if (arr.ndim() != Type::NumIndices) {
            return false;
        }

	    Eigen::DSizes<typename Type::Index, Type::NumIndices> shape;
        std::copy(arr.shape(), arr.shape() + Type::NumIndices, shape.begin());

        if (!Helper::is_correct_shape(shape)) {
            return false;
        }

        value = Eigen::TensorMap<Type>(const_cast<typename Type::Scalar *>(arr.data()), shape);

        return true;
    }

    static handle cast(Type &&src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::reference
            || policy == return_value_policy::reference_internal) {
            pybind11_fail("Cannot use a reference return value policy for an rvalue");
        }
        return cast_impl(&src, return_value_policy::move, parent);
    }

    static handle cast(const Type &&src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::reference
            || policy == return_value_policy::reference_internal) {
            pybind11_fail("Cannot use a reference return value policy for an rvalue");
        }
        return cast_impl(&src, return_value_policy::move, parent);
    }

    static handle cast(Type &src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic
            || policy == return_value_policy::automatic_reference) {
            policy = return_value_policy::copy;
        }
        return cast_impl(&src, policy, parent);
    }

    static handle cast(const Type &src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic
            || policy == return_value_policy::automatic_reference) {
            policy = return_value_policy::copy;
        }
        return cast(&src, policy, parent);
    }

    static handle cast(Type *src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic) {
            policy = return_value_policy::take_ownership;
        } else if (policy == return_value_policy::automatic_reference) {
            policy = return_value_policy::reference;
        }
        return cast_impl(src, policy, parent);
    }

    static handle cast(const Type *src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic) {
            policy = return_value_policy::take_ownership;
        } else if (policy == return_value_policy::automatic_reference) {
            policy = return_value_policy::reference;
        }
        return cast_impl(src, policy, parent);
    }

    template <typename C>
    static handle cast_impl(C *src, return_value_policy policy, handle parent) {
        object parent_object;
        bool writeable = false;
        switch (policy) {
            case return_value_policy::move:
                if (std::is_const<C>::value) {
                    pybind11_fail("Cannot move from a constant reference");
                }

                src = Helper::alloc(std::move(*src));

                parent_object
                    = capsule(src, [](void *ptr) { Helper::free(reinterpret_cast<Type *>(ptr)); });
                writeable = true;
                break;

            case return_value_policy::take_ownership:
                if (std::is_const<C>::value) {
                    pybind11_fail("Cannot take ownership of a const reference");
                }

                parent_object
                    = capsule(src, [](void *ptr) { Helper::free(reinterpret_cast<Type *>(ptr)); });
                writeable = true;
                break;

            case return_value_policy::copy: {
                // Clang 3.6 / 3.7 has a very confusing bug that seems to be fixed by adding this
                // scope
                parent_object = {};
            }
                writeable = true;
                break;

            case return_value_policy::reference:
                parent_object = none();
                writeable = !std::is_const<C>::value;
                break;

            case return_value_policy::reference_internal:
                // Default should do the right thing
                if (!parent) {
                    pybind11_fail("Cannot use reference internal when there is no parent");
                }
                parent_object = reinterpret_borrow<object>(parent);
                writeable = !std::is_const<C>::value;
                break;

            default:
                pybind11_fail("pybind11 bug in eigen.h, please file a bug report");
        }

        auto result = array_t<typename Type::Scalar, compute_array_flag_from_tensor<Type>()>(
            Helper::get_shape(*src), src->data(), parent_object);

        if (!writeable) {
            array_proxy(result.ptr())->flags &= ~detail::npy_api::NPY_ARRAY_WRITEABLE_;
        }

        return result.release();
    }
};

template <typename Type>
struct type_caster<Eigen::TensorMap<Type>, typename eigen_tensor_helper<Type>::ValidType> {
    using Helper = eigen_tensor_helper<Type>;

    bool load(handle src, bool /*convert*/) {
        // Note that we have a lot more checks here as we want to make sure to avoid copies
        auto arr = reinterpret_borrow<array>(src);
        if ((arr.flags() & compute_array_flag_from_tensor<Type>()) == 0) {
            return false;
        }

        if (!arr.dtype().is(dtype::of<typename Type::Scalar>())) {
            return false;
        }

        if (arr.ndim() != Type::NumIndices) {
            return false;
        }

        Eigen::DSizes<typename Type::Index, Type::NumIndices> shape;
        std::copy(arr.shape(), arr.shape() + Type::NumIndices, shape.begin());

        if (!Helper::is_correct_shape(shape)) {
            return false;
        }

        value.reset(new Eigen::TensorMap<Type>(
            reinterpret_cast<typename Type::Scalar *>(arr.mutable_data()), shape));

        return true;
    }

    static handle cast(Eigen::TensorMap<Type> &&src, return_value_policy policy, handle parent) {
        return cast_impl(&src, policy, parent);
    }

    static handle
    cast(const Eigen::TensorMap<Type> &&src, return_value_policy policy, handle parent) {
        return cast_impl(&src, policy, parent);
    }

    static handle cast(Eigen::TensorMap<Type> &src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic
            || policy == return_value_policy::automatic_reference) {
            policy = return_value_policy::copy;
        }
        return cast_impl(&src, policy, parent);
    }

    static handle
    cast(const Eigen::TensorMap<Type> &src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic
            || policy == return_value_policy::automatic_reference) {
            policy = return_value_policy::copy;
        }
        return cast(&src, policy, parent);
    }

    static handle cast(Eigen::TensorMap<Type> *src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic) {
            policy = return_value_policy::take_ownership;
        } else if (policy == return_value_policy::automatic_reference) {
            policy = return_value_policy::reference;
        }
        return cast_impl(src, policy, parent);
    }

    static handle
    cast(const Eigen::TensorMap<Type> *src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic) {
            policy = return_value_policy::take_ownership;
        } else if (policy == return_value_policy::automatic_reference) {
            policy = return_value_policy::reference;
        }
        return cast_impl(src, policy, parent);
    }

    template <typename C>
    static handle cast_impl(C *src, return_value_policy policy, handle parent) {
        object parent_object;
        constexpr bool writeable = !std::is_const<C>::value;
        switch (policy) {
            case return_value_policy::reference:
                parent_object = none();
                break;

            case return_value_policy::reference_internal:
                // Default should do the right thing
                if (!parent) {
                    pybind11_fail("Cannot use reference internal when there is no parent");
                }
                parent_object = reinterpret_borrow<object>(parent);
                break;

            default:
                // move, take_ownership don't make any sense for a ref/map:
                pybind11_fail("Invalid return_value_policy for Eigen Map type, must be either "
                              "reference or reference_internal");
        }

        auto result = array_t<typename Type::Scalar, compute_array_flag_from_tensor<Type>()>(
            Helper::get_shape(*src), src->data(), parent_object);

        if (!writeable) {
            array_proxy(result.ptr())->flags &= ~detail::npy_api::NPY_ARRAY_WRITEABLE_;
        }

        return result.release();
    }

protected:
    // TODO: Move to std::optional once std::optional has more support
    std::unique_ptr<Eigen::TensorMap<Type>> value;

public:
    static constexpr auto name = get_tensor_descriptor<Type>::value;
    explicit operator Eigen::TensorMap<Type> *() { return value.get(); }
    explicit operator Eigen::TensorMap<Type> &() { return *value; }
    explicit operator Eigen::TensorMap<Type> &&() && { return std::move(*value); }

    template <typename T_>
    using cast_op_type = ::pybind11::detail::movable_cast_op_type<T_>;
};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
