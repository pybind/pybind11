// Copyright (c) 2026 The pybind Community.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <fuzzer/FuzzedDataProvider.h>

namespace pybind11_fuzz {

struct input {
    std::string code;
    std::string attr_name;
    std::string decimal_text;
    std::string decimal_attr;
    std::string text;
};

inline input decode_input(const std::uint8_t *data, std::size_t size) {
    FuzzedDataProvider fdp(data, size);
    input result;
    result.code = fdp.ConsumeRandomLengthString();
    result.attr_name = fdp.ConsumeRandomLengthString();
    result.decimal_text = fdp.ConsumeRandomLengthString();
    result.decimal_attr = fdp.ConsumeRandomLengthString();
    result.text = fdp.ConsumeRandomLengthString();
    return result;
}

} // namespace pybind11_fuzz
