#pragma once

#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

namespace pybind11_ubench {

template <int Serial>
struct number_bucket {
    std::vector<double> data;

    explicit number_bucket(std::size_t data_size = 0) : data(data_size, 1.0) {}

    double sum() const {
        std::size_t n = 0;
        double s = 0;
        const double *a = &*data.begin();
        const double *e = &*data.end();
        while (a != e) {
            s += *a++;
            n++;
        }
        if (n != data.size()) {
            std::cerr << "Internal consistency failure (sum)." << std::endl;
            std::terminate();
        }
        return s;
    }

    std::size_t add(const number_bucket &other) {
        if (other.data.size() != data.size()) {
            std::cerr << "Incompatible data sizes (add)." << std::endl;
            std::terminate();
        }
        std::size_t n = 0;
        double *a = &*data.begin();
        const double *e = &*data.end();
        const double *b = &*other.data.begin();
        while (a != e) {
            *a++ += *b++;
            n++;
        }
        return n;
    }

private:
    number_bucket(const number_bucket &) = delete;
    number_bucket(number_bucket &&) = delete;
    number_bucket &operator=(const number_bucket &) = delete;
    number_bucket &operator=(number_bucket &&) = delete;
};

} // namespace pybind11_ubench
