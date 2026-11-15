// Copyright (c) 2026 The pybind Community.

#include "fuzz_input.h"

#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <exec_success corpus input>\n";
        return 2;
    }
    std::ifstream file(argv[1], std::ios::binary);
    if (!file) {
        std::cerr << "failed to open fuzz input: " << argv[1] << '\n';
        return 1;
    }
    const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(file),
                                          std::istreambuf_iterator<char>()};
    const auto input = pybind11_fuzz::decode_input(bytes.data(), bytes.size());
    if (input.code != "x = 1" || input.attr_name != "getcwd" || input.decimal_text != "1"
        || input.decimal_attr != "as_tuple" || input.text != "ok") {
        std::cerr << "unexpected fields decoded from exec_success corpus input\n";
        return 1;
    }
    return 0;
}
