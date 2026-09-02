#include "pybind11/pybind11.h"
#include "catch.hpp"

namespace py = pybind11;

// 2 is chosen because it is the smallest number (keeping tests short)
// where we can create non-empty vectors whose size is the inline size
// plus or minus 1.
using argument_vector = py::detail::argument_vector<2>;

namespace {
argument_vector to_argument_vector(const std::vector<py::handle> &v) {
    argument_vector result;
    result.reserve(v.size());
    for (const auto x : v) {
        result.push_back(x);
    }
    return result;
}

std::vector<std::vector<py::handle>> get_sample_argument_vector_contents() {
    return std::vector<std::vector<py::handle>>{
        {},
        {py::handle(Py_None)},
        {py::handle(Py_None), py::handle(Py_False)},
        {py::handle(Py_None), py::handle(Py_False), py::handle(Py_True)},
    };
}

std::vector<argument_vector> get_sample_argument_vectors() {
    std::vector<argument_vector> result;
    for (const auto &vec : get_sample_argument_vector_contents()) {
        result.push_back(to_argument_vector(vec));
    }
    return result;
}

void require_vector_matches_sample(const argument_vector &actual,
                                   const std::vector<py::handle> &expected) {
    REQUIRE(actual.size() == expected.size());
    for (size_t ii = 0; ii < actual.size(); ++ii) {
        REQUIRE(actual[ii].ptr() == expected[ii].ptr());
    }
}

template <typename ActualMutationFunc, typename ExpectedMutationFunc>
void mutation_test_with_samples(ActualMutationFunc actual_mutation_func,
                                ExpectedMutationFunc expected_mutation_func) {
    auto sample_contents = get_sample_argument_vector_contents();
    auto samples = get_sample_argument_vectors();
    for (size_t ii = 0; ii < samples.size(); ++ii) {
        auto &actual = samples[ii];
        auto &expected = sample_contents[ii];

        actual_mutation_func(actual);
        expected_mutation_func(expected);
        require_vector_matches_sample(actual, expected);
    }
}

} // namespace

// I would like to write [capture](auto& vec) block inline, but we
// have to work with C++11, which doesn't have generic lambdas.
// NOLINTBEGIN(bugprone-macro-parentheses)
#define MUTATION_LAMBDA(capture, block)                                                           \
    [capture](argument_vector & vec) block, [capture](std::vector<py::handle> & vec) block
// NOLINTEND(bugprone-macro-parentheses)

// For readability, rather than having ugly empty arguments.
#define NO_CAPTURE

TEST_CASE("check sample argument_vector contents") {
    mutation_test_with_samples(MUTATION_LAMBDA(NO_CAPTURE, { (void) vec; }));
}

TEST_CASE("argument_vector push_back") {
    mutation_test_with_samples(MUTATION_LAMBDA(NO_CAPTURE, { vec.emplace_back(Py_None); }));
}

TEST_CASE("argument_vector reserve") {
    for (std::size_t ii = 0; ii < 4; ++ii) {
        mutation_test_with_samples(MUTATION_LAMBDA(ii, { vec.reserve(ii); }));
    }
}

TEST_CASE("argument_vector reserve then push_back") {
    for (std::size_t ii = 0; ii < 4; ++ii) {
        mutation_test_with_samples(MUTATION_LAMBDA(ii, {
            vec.reserve(ii);
            vec.emplace_back(Py_True);
        }));
    }
}

namespace {

// Extra references so that a buggy double-decref cannot free the object under
// test and turn a refcount mismatch into a crash.
constexpr int kRefPadding = 100;

using ref_small_vector = py::detail::ref_small_vector<2>;

void fill_with_borrowed(ref_small_vector &vec, PyObject *obj, std::size_t count) {
    for (std::size_t ii = 0; ii < count; ++ii) {
        vec.push_back_borrow(obj);
    }
}

void check_move_construct(std::size_t count) {
    py::list obj;
    for (int ii = 0; ii < kRefPadding; ++ii) {
        Py_INCREF(obj.ptr());
    }
    const auto initial = obj.ref_count();
    {
        ref_small_vector src;
        fill_with_borrowed(src, obj.ptr(), count);
        REQUIRE(obj.ref_count() == initial + static_cast<py::ssize_t>(count));

        ref_small_vector dst(std::move(src));
        REQUIRE(dst.size() == count);
        // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
        REQUIRE(src.size() == 0);
        REQUIRE(obj.ref_count() == initial + static_cast<py::ssize_t>(count));
    }
    REQUIRE(obj.ref_count() == initial);
    for (int ii = 0; ii < kRefPadding; ++ii) {
        Py_DECREF(obj.ptr());
    }
}

void check_move_assign(std::size_t count) {
    py::list obj;
    for (int ii = 0; ii < kRefPadding; ++ii) {
        Py_INCREF(obj.ptr());
    }
    const auto initial = obj.ref_count();
    {
        ref_small_vector src;
        fill_with_borrowed(src, obj.ptr(), count);
        ref_small_vector dst;
        fill_with_borrowed(dst, obj.ptr(), 1);
        dst = std::move(src);
        REQUIRE(dst.size() == count);
        // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
        REQUIRE(src.size() == 0);
        REQUIRE(obj.ref_count() == initial + static_cast<py::ssize_t>(count));
    }
    REQUIRE(obj.ref_count() == initial);
    for (int ii = 0; ii < kRefPadding; ++ii) {
        Py_DECREF(obj.ptr());
    }
}

} // namespace

TEST_CASE("ref_small_vector move construction releases each reference once") {
    // 0..2 use the inline array, 3 and 5 use the heap vector.
    for (std::size_t count :
         {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(3), std::size_t(5)}) {
        check_move_construct(count);
    }
}

TEST_CASE("ref_small_vector move assignment releases each reference once") {
    for (std::size_t count :
         {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(3), std::size_t(5)}) {
        check_move_assign(count);
    }
}
