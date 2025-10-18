#include "pybind11/pybind11.h"
#include "catch.hpp"

namespace py = pybind11;

using args_convert_vector = py::detail::args_convert_vector<py::detail::arg_vector_small_size>;

namespace {
template <typename Container>
std::vector<Container> get_sample_vectors() {
    std::vector<Container> result;
    result.emplace_back();
    for (const auto sz : {0, 4, 5, 6, 31, 32, 33, 63, 64, 65}) {
        for (const bool b : {false, true}) {
            result.emplace_back(static_cast<std::size_t>(sz), b);
        }
    }
    return result;
}

void require_vector_matches_sample(const args_convert_vector &actual,
                                   const std::vector<bool> &expected) {
    REQUIRE(actual.size() == expected.size());
    for (size_t ii = 0; ii < actual.size(); ++ii) {
        REQUIRE(actual[ii] == expected[ii]);
    }
}

template <typename ActualMutationFunc, typename ExpectedMutationFunc>
void mutation_test_with_samples(ActualMutationFunc actual_mutation_func,
                                ExpectedMutationFunc expected_mutation_func) {
    auto sample_contents = get_sample_vectors<std::vector<bool>>();
    auto samples = get_sample_vectors<args_convert_vector>();
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
    [capture](args_convert_vector & vec) block, [capture](std::vector<bool> & vec) block
// NOLINTEND(bugprone-macro-parentheses)

// For readability, rather than having ugly empty arguments.
#define NO_CAPTURE

TEST_CASE("check sample args_convert_vector contents") {
    mutation_test_with_samples(MUTATION_LAMBDA(NO_CAPTURE, { (void) vec; }));
}

TEST_CASE("args_convert_vector push_back") {
    for (const bool b : {false, true}) {
        mutation_test_with_samples(MUTATION_LAMBDA(b, { vec.push_back(b); }));
    }
}

TEST_CASE("args_convert_vector reserve") {
    for (std::size_t ii = 0; ii < 4; ++ii) {
        mutation_test_with_samples(MUTATION_LAMBDA(ii, { vec.reserve(ii); }));
    }
}

TEST_CASE("args_convert_vector reserve then push_back") {
    for (std::size_t ii = 0; ii < 4; ++ii) {
        for (const bool b : {false, true}) {
            mutation_test_with_samples(MUTATION_LAMBDA(=, {
                vec.reserve(ii);
                vec.push_back(b);
            }));
        }
    }
}
