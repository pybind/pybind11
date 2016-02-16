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

/*
	WARNING: These iterators are not a binding to numpy.nditer, they're convenient classes for broadcasting in vectorize
*/

NAMESPACE_BEGIN(pybind11)

template  <class T>
using array_iterator = typename std::add_pointer<T>::type;

template <class T>
array_iterator<T> array_begin(const buffer_info& buffer)
{
	return array_iterator<T>(reinterpret_cast<T*>(buffer.ptr));
}

template <class T>
array_iterator<T> array_end(const buffer_info& buffer)
{
	return array_iterator<T>(reinterpret_cast<T*>(buffer.ptr) + buffer.size);
}

NAMESPACE_BEGIN(detail)

template <class C>
class common_iterator
{

public:

	using container_type = C;
	using value_type = typename container_type::value_type;
	using size_type = typename container_type::size_type;

	common_iterator() : p_ptr(0), m_strides() {}
	common_iterator(void* ptr, const container_type& strides, const std::vector<size_t>& shape)
		: p_ptr(reinterpret_cast<char*>(ptr)), m_strides(strides.size())
	{
		m_strides.back() = static_cast<value_type>(strides.back());
		for (size_type i = m_strides.size() - 1; i != 0; --i)
		{
			size_type j = i - 1;
			value_type s = static_cast<value_type>(shape[i]);
			m_strides[j] = strides[j] + m_strides[i] - strides[i] * s;
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
	container_type m_strides;
};

NAMESPACE_END(detail)

template <class C, size_t N>
class multi_array_iterator
{

public:

	using container_type = C;

	multi_array_iterator(const std::array<buffer_info, N>& buffers,
						 const std::vector<size_t>& shape)
		: m_shape(shape.size()), m_index(shape.size(), 0), m_common_iterator()
	{
		// Manual copy to avoid conversion warning if using std::copy
		for (size_t i = 0; i < shape.size(); ++i)
		{
			m_shape[i] = static_cast<typename container_type::value_type>(shape[i]);
		}

		container_type strides(shape.size());
		for (size_t i = 0; i < N; ++i)
		{
			init_common_iterator(buffers[i], shape, m_common_iterator[i], strides);
		}
	}

	multi_array_iterator& operator++()
	{
		for (size_t j = m_index.size(); j != 0; --j)
		{
			size_t i = j - 1;
			if (++m_index[i] != m_shape[i])
			{
				increment_common_iterator(i);
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

	using common_iter = detail::common_iterator<container_type>;

	void init_common_iterator(const buffer_info& buffer, const std::vector<size_t>& shape, common_iter& iterator, container_type& strides)
	{
		auto buffer_shape_iter = buffer.shape.rbegin();
		auto buffer_strides_iter = buffer.strides.rbegin();
		auto shape_iter = shape.rbegin();
		auto strides_iter = strides.rbegin();

		while (buffer_shape_iter != buffer.shape.rend())
		{
			if (*shape_iter == *buffer_shape_iter)
				*strides_iter = static_cast<int>(*buffer_strides_iter);
			else
				*strides_iter = 0;

			++buffer_shape_iter;
			++buffer_strides_iter;
			++shape_iter;
			++strides_iter;
		}

		std::fill(strides_iter, strides.rend(), 0);

		iterator = common_iter(buffer.ptr, strides, shape);
	}

	void increment_common_iterator(size_t dim)
	{
		std::for_each(m_common_iterator.begin(), m_common_iterator.end(), [=](common_iter& iter)
		{
			iter.increment(dim);
		});
	}

	container_type m_shape;
	container_type m_index;
	std::array<common_iter, N> m_common_iterator;
};

NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
