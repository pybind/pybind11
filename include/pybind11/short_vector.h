/*
	pybind11/short_vector.h: similar to std::array but with dynamic size

	Copyright (c) 2016 Johan Mabille

	All rights reserved. Use of this source code is governed by a
	BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"
#include <array>
#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#endif

NAMESPACE_BEGIN(pybind11)

template <class T, size_t N = 3>
class short_vector
{

public:

	using data_type = std::array<T, N>;
	using value_type = typename data_type::value_type;
	using size_type = typename data_type::size_type;
	using difference_type = typename data_type::difference_type;
	using reference = typename data_type::reference;
	using const_reference = typename data_type::const_reference;
	using pointer = typename data_type::pointer;
	using const_pointer = typename data_type::const_pointer;
	using iterator = typename data_type::iterator;
	using const_iterator = typename data_type::const_iterator;
	using reverse_iterator = typename data_type::reverse_iterator;
	using const_reverse_iterator = typename data_type::const_reverse_iterator;

	short_vector() : m_data(), m_size(0) {}
	explicit short_vector(size_type size) : m_data(), m_size(size) {}
	short_vector(size_type size, const_reference t) : m_data(), m_size(size)
	{
		std::fill(begin(), end(), t);
	}

	size_type size() const noexcept { return m_size; }
	constexpr size_type max_size() const noexcept { return m_data.max_size(); }
	bool empty() const noexcept { return m_size == 0; }

	void resize(size_type size) { m_size = size; }
	void resize(size_type size, const_reference t)
	{
		size_type old_size = m_size;
		resize(size); std::fill(begin() + old_size, end(), t);
	}

	reference operator[](size_type i) { return m_data[i]; }
	const_reference operator[](size_type i) const { return m_data[i]; }

	reference front() { return m_data[0]; }
	const_reference front() const { return m_data[0]; }

	reference back() { return m_data[m_size - 1]; }
	const_reference back() const { return m_data[m_size - 1]; }

	void fill(const_reference t) { std::fill(begin(), end(), t); }

	iterator begin() noexcept { return m_data.begin(); }
	const_iterator begin() const noexcept { return m_data.begin(); }
	iterator end() noexcept { return begin() + m_size; }
	const_iterator end() const noexcept { return begin() + m_size(); }

	reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
	const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
	reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
	const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }

	const_iterator cbegin() const noexcept { return begin(); }
	const_iterator cend() const noexcept { return end(); }
	const_reverse_iterator crbegin() const noexcept { return rbegin(); }
	const_reverse_iterator crend() const noexcept { return rend(); }

private:

	data_type m_data;
	size_type m_size;

};

NAMESPACE_END(pybind11)

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
