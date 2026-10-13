// Local and CI driver for LLVMFuzzerTestOneInput. OSS-Fuzz supplies its own
// main() through LIB_FUZZING_ENGINE, so this file is not linked there.

#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

namespace {

std::vector<uint8_t>
read_file(const std::string &path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open fuzz input: " + path);
    }
    input.seekg(0, std::ios::end);
    const auto nbytes = input.tellg();
    if (nbytes < 0) {
        throw std::runtime_error("failed to size fuzz input: " + path);
    }
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(nbytes));
    if (nbytes > 0) {
        input.read(reinterpret_cast<char *>(bytes.data()), nbytes);
    }
    if (!input) {
        throw std::runtime_error("failed to read fuzz input: " + path);
    }
    return bytes;
}

} // namespace

int
main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <input>...\n";
        return 2;
    }
    try {
        for (int i = 1; i < argc; ++i) {
            const std::vector<uint8_t> bytes = read_file(argv[i]);
            const int rc = LLVMFuzzerTestOneInput(bytes.data(), bytes.size());
            if (rc != 0) {
                std::cerr << "fuzz callback failed on " << argv[i] << "\n";
                return rc;
            }
        }
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << "\n";
        return 1;
    }
    return 0;
}
