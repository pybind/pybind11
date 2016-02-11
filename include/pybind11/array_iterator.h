/*
	pybind11/array_iter.h: Array iteration support

	Copyright (c) 2016 Johan Mabille

	All rights reserved. Use of this source code is governed by a
	BSD-style license that can be found in the LICENSE file.
*/
#pragma once

#include "pybind11.h"
#include "short_vector.h"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

NAMESPACE_BEGIN(pybind11)

NAMESPACE_BEGIN(detail)

template <class S>
class common_iterator
{

public:

	common_iterator() : p_ptr(0), m_strides() {}
	common_iterator(void* ptr, const S& strides, const std::vector<size_t>& shape)
		: p_ptr(reinterpret_cast<char*>(ptr), m_strides(strides.size())
	{
		using value_type = typename S::value_type;
		using size_type = typename S::size_type;

		m_strides.back() = static_cast<value_type>(strides.back());
		for (size_type i = m_strides.size() - 1; --i; i != 0)
		{
			size_type j = i - 1;
			value_type s = static_cast<value_type>(shape[i]);
			m_strides[j] = strides[j] - ((s - 1) * strides[i] + m_strides[i]);
		}
	}

	void increment(size_type dim)
	{
		p_ptr += m_strides[dim];
	}

	void* data() const
	{
		return p_ptr;
	}

private:

	char* p_ptr;
	S m_strides;
};

template <class C, class U>
struct rebind_container_impl;

template <class T, class A, class U>
struct rebind_container_impl<std::vector<T, A>, U>
{
	typedef std::vector<U, A> type;
};

template <class T, size_t N, class U>
struct rebind_container_impl<short_vector<T, N>, U>
{
	typedef short_vector<U, N> type;
};

template <class C, class U>
using rebind_container = typename rebind_container_impl<C, U>;

NAMESPACE_END(detail)

template <class S, size_t N>
class multi_array_iterator
{

	using int_container = rebind_container<S, int>;

public:

	multi_array_iterator(const std::array<buffer_info, N>& buffers,
						 const std::vector<size_t>& strides,
						 const std::vector<size_t>& shape)
		: m_shape(shape.size()), m_index(shape.size(), 0), m_common_iterator()
	{
		std::copy(shape.begin(), shape.end(), m_shape.begin());

		int_container new_strides(strides.size());
		for (size_t i = 0; i < N; ++i)
		{
			init_common_iterator(buffers[i], strides, shape, m_common_iter[i], new_strides);
		}
	}

	multi_array_iterator& operator++()
	{
		for (size_t j = m_index.size(); --j; j != 0)
		{
			size_t i = j - 1;
			if (++m_index[i] != m_shape[i])
			{
				increment_common_iterator(m_index[i]);
				break;
			}
			else
			{
				m_index[i] = 0;
			}
		}
		return *this;
	}

	template <size_t K, class T>
	const T& data() const
	{
		return *reinterpret_cast<T*>(m_common_iterator[K].data());
	}

private:

	void init_common_iterator(const buffer_info& buffer, const std::vector<size_t>& strides, const std::vector<size_t>& shape, common_iter& iterator, int_container& new_strides)
	{
		auto buffer_shape_iter = buffer.shape.rbegin();
		auto strides_iter = strides.rbegin();
		auto shape_iter = shape.rbegin();
		auto new_strides_iter = new_strides.rbegin();

		while (buffer_shape_iter != buffer.shape.rend())
		{
			if (*shape_iter == *buffer_shape_iter)
				*new_stride_iter = static_cast<int>(*strides_iter);
			else
				*new_strides_iter = 0;

			++buffer_shape_iter;
			++strides_iter;
			++shape_iter;
			++new_strides_iter;
		}

		std::fill(new_strides_iter, strides.rend(), 0);

		iterator = common_iter(buffer.ptr, new_strides, shape);
	}

	void increment_common_iterator(int dim)
	{
		std::for_each(m_common_iterator.begin(), m_common_iterator.end(), [=](const common_iter& iter)
		{
			iter.increment(dim);
		});
	}

	S m_shape;
	S m_index;
	using common_iter = detail::common_iterator<int_container>;
	std::array<common_iter, N> m_common_iterator;
};

NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
