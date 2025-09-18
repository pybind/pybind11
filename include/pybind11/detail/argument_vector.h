/*
    pybind11/detail/argument_vector.h: small_vector-like containers to
    avoid heap allocation of arguments during function call dispatch.

    Copyright (c) Meta Platforms, Inc. and affiliates.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include <pybind11/pytypes.h>

#include "common.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_WARNING_DISABLE_MSVC(4127)

PYBIND11_NAMESPACE_BEGIN(detail)

// Shared implementation utility for our small_vector-like containers.
// We support C++11 and C++14, so we cannot use
// std::variant. Union with the tag packed next to the inline
// array's size is smaller anyway, allowing 1 extra handle of
// inline storage for free. Compare the layouts (1 line per
// size_t/void*, assuming a 64-bit machine):
// With variant, total is N + 2 for N >= 2:
// - variant tag (cannot be packed with the array size)
// - array size (or first pointer of 3 in std::vector)
// - N pointers of inline storage (or 2 remaining pointers of std::vector)
// Custom union, total is N + 1 for N >= 3:
// - variant tag & array size if applicable
// - N pointers of inline storage (or 3 pointers of std::vector)
//
// NOTE: this is a low-level representational convenience; the two
// use cases of this union are materially different and in particular
// have different semantics for inline_array::size. All that is being
// shared is the memory management behavior.
template <typename ArrayT, std::size_t InlineSize, typename VectorT = ArrayT>
union inline_array_or_vector {
    struct inline_array {
        bool is_inline = true;
        std::uint32_t size = 0;
        std::array<ArrayT, InlineSize> arr;
    };
    struct heap_vector {
        bool is_inline = false;
        std::vector<VectorT> vec;

        heap_vector() = default;
        heap_vector(std::size_t count, VectorT value) : vec(count, value) {}
    };

    inline_array iarray;
    heap_vector hvector;

    static_assert(std::is_trivially_move_constructible<ArrayT>::value,
                  "ArrayT must be trivially move constructible");
    static_assert(std::is_trivially_destructible<ArrayT>::value,
                  "ArrayT must be trivially destructible");

    inline_array_or_vector() : iarray() {}
    ~inline_array_or_vector() {
        if (!is_inline()) {
            hvector.~heap_vector();
        }
    }
    // Disable copy ctor and assignment.
    inline_array_or_vector(const inline_array_or_vector &) = delete;
    inline_array_or_vector &operator=(const inline_array_or_vector &) = delete;

    inline_array_or_vector(inline_array_or_vector &&rhs) noexcept {
        if (rhs.is_inline()) {
            std::memcpy(&iarray, &rhs.iarray, sizeof(iarray));
        } else {
            new (&hvector) heap_vector(std::move(rhs.hvector));
        }
        assert(is_inline() == rhs.is_inline());
    }

    inline_array_or_vector &operator=(inline_array_or_vector &&rhs) noexcept {
        if (this == &rhs) {
            return *this;
        }

        if (rhs.is_inline()) {
            if (!is_inline()) {
                hvector.~heap_vector();
            }
            std::memcpy(&iarray, &rhs.iarray, sizeof(iarray));
        } else {
            if (is_inline()) {
                new (&hvector) heap_vector(std::move(rhs.hvector));
            } else {
                hvector = std::move(rhs.hvector);
            }
        }
        return *this;
    }

    bool is_inline() const {
        // It is undefined behavior to access the inactive member of a
        // union directly. However, it is well-defined to reinterpret_cast any
        // pointer into a pointer to char and examine it as an array
        // of bytes. See
        // https://dev-discuss.pytorch.org/t/unionizing-for-profit-how-to-exploit-the-power-of-unions-in-c/444#the-memcpy-loophole-4
        bool result = false;
        static_assert(offsetof(inline_array, is_inline) == 0,
                      "untagged union implementation relies on this");
        static_assert(offsetof(heap_vector, is_inline) == 0,
                      "untagged union implementation relies on this");
        std::memcpy(&result, reinterpret_cast<const char *>(this), sizeof(bool));
        return result;
    }
};

// small_vector-like container to avoid heap allocation for N or fewer
// arguments.
template <std::size_t N>
struct argument_vector {
public:
    argument_vector() = default;

    // Disable copy ctor and assignment.
    argument_vector(const argument_vector &) = delete;
    argument_vector &operator=(const argument_vector &) = delete;
    argument_vector(argument_vector &&) noexcept = default;
    argument_vector &operator=(argument_vector &&) noexcept = default;

    std::size_t size() const {
        if (is_inline()) {
            return m_repr.iarray.size;
        }
        return m_repr.hvector.vec.size();
    }

    handle &operator[](std::size_t idx) {
        assert(idx < size());
        if (is_inline()) {
            return m_repr.iarray.arr[idx];
        }
        return m_repr.hvector.vec[idx];
    }

    handle operator[](std::size_t idx) const {
        assert(idx < size());
        if (is_inline()) {
            return m_repr.iarray.arr[idx];
        }
        return m_repr.hvector.vec[idx];
    }

    void push_back(handle x) {
        if (is_inline()) {
            auto &ha = m_repr.iarray;
            if (ha.size == N) {
                move_to_heap_vector_with_reserved_size(N + 1);
                push_back_slow_path(x);
            } else {
                ha.arr[ha.size++] = x;
            }
        } else {
            push_back_slow_path(x);
        }
    }

    template <typename Arg>
    void emplace_back(Arg &&x) {
        push_back(handle(x));
    }

    void reserve(std::size_t sz) {
        if (is_inline()) {
            if (sz > N) {
                move_to_heap_vector_with_reserved_size(sz);
            }
        } else {
            reserve_slow_path(sz);
        }
    }

private:
    using repr_type = inline_array_or_vector<handle, N>;
    repr_type m_repr;

    PYBIND11_NOINLINE void move_to_heap_vector_with_reserved_size(std::size_t reserved_size) {
        assert(is_inline());
        auto &ha = m_repr.iarray;
        using heap_vector = typename repr_type::heap_vector;
        heap_vector hv;
        hv.vec.reserve(reserved_size);
        std::copy(ha.arr.begin(), ha.arr.begin() + ha.size, std::back_inserter(hv.vec));
        new (&m_repr.hvector) heap_vector(std::move(hv));
    }

    PYBIND11_NOINLINE void push_back_slow_path(handle x) { m_repr.hvector.vec.push_back(x); }

    PYBIND11_NOINLINE void reserve_slow_path(std::size_t sz) { m_repr.hvector.vec.reserve(sz); }

    bool is_inline() const { return m_repr.is_inline(); }
};

// small_vector-like container to avoid heap allocation for N or fewer
// arguments.
template <std::size_t kRequestedInlineSize>
struct args_convert_vector {
private:
public:
    args_convert_vector() = default;

    // Disable copy ctor and assignment.
    args_convert_vector(const args_convert_vector &) = delete;
    args_convert_vector &operator=(const args_convert_vector &) = delete;
    args_convert_vector(args_convert_vector &&) noexcept = default;
    args_convert_vector &operator=(args_convert_vector &&) noexcept = default;

    args_convert_vector(std::size_t count, bool value) {
        if (count > kInlineSize) {
            new (&m_repr.hvector) typename repr_type::heap_vector(count, value);
        } else {
            auto &inline_arr = m_repr.iarray;
            inline_arr.arr.fill(value ? std::size_t(-1) : 0);
            inline_arr.size = static_cast<decltype(inline_arr.size)>(count);
        }
    }

    std::size_t size() const {
        if (is_inline()) {
            return m_repr.iarray.size;
        }
        return m_repr.hvector.vec.size();
    }

    void reserve(std::size_t sz) {
        if (is_inline()) {
            if (sz > kInlineSize) {
                move_to_heap_vector_with_reserved_size(sz);
            }
        } else {
            m_repr.hvector.vec.reserve(sz);
        }
    }

    bool operator[](std::size_t idx) const {
        if (is_inline()) {
            return inline_index(idx);
        }
        assert(idx < m_repr.hvector.vec.size());
        return m_repr.hvector.vec[idx];
    }

    void push_back(bool b) {
        if (is_inline()) {
            auto &ha = m_repr.iarray;
            if (ha.size == kInlineSize) {
                move_to_heap_vector_with_reserved_size(kInlineSize + 1);
                push_back_slow_path(b);
            } else {
                assert(ha.size < kInlineSize);
                const auto wbi = word_and_bit_index(ha.size++);
                assert(wbi.word < kWords);
                assert(wbi.bit < kBitsPerWord);
                if (b) {
                    ha.arr[wbi.word] |= (std::size_t(1) << wbi.bit);
                } else {
                    ha.arr[wbi.word] &= ~(std::size_t(1) << wbi.bit);
                }
                assert(operator[](ha.size - 1) == b);
            }
        } else {
            push_back_slow_path(b);
        }
    }

    void swap(args_convert_vector &rhs) noexcept { std::swap(m_repr, rhs.m_repr); }

private:
    struct WordAndBitIndex {
        std::size_t word;
        std::size_t bit;
    };

    static WordAndBitIndex word_and_bit_index(std::size_t idx) {
        return WordAndBitIndex{idx / kBitsPerWord, idx % kBitsPerWord};
    }

    bool inline_index(std::size_t idx) const {
        const auto wbi = word_and_bit_index(idx);
        assert(wbi.word < kWords);
        assert(wbi.bit < kBitsPerWord);
        return m_repr.iarray.arr[wbi.word] & (std::size_t(1) << wbi.bit);
    }

    PYBIND11_NOINLINE void move_to_heap_vector_with_reserved_size(std::size_t reserved_size) {
        auto &inline_arr = m_repr.iarray;
        using heap_vector = typename repr_type::heap_vector;
        heap_vector hv;
        hv.vec.reserve(reserved_size);
        for (std::size_t ii = 0; ii < inline_arr.size; ++ii) {
            hv.vec.push_back(inline_index(ii));
        }
        new (&m_repr.hvector) heap_vector(std::move(hv));
    }

    PYBIND11_NOINLINE void push_back_slow_path(bool b) { m_repr.hvector.vec.push_back(b); }

    static constexpr auto kBitsPerWord = 8 * sizeof(std::size_t);
    static constexpr auto kWords = (kRequestedInlineSize + kBitsPerWord - 1) / kBitsPerWord;
    static constexpr auto kInlineSize = kWords * kBitsPerWord;

    using repr_type = inline_array_or_vector<std::size_t, kWords, bool>;
    repr_type m_repr;

    bool is_inline() const { return m_repr.is_inline(); }
};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
