// Links a pybind11_add_module(... SHARED ...) library from an executable.
extern "C" void *PyInit_test_cmake_build();

int main(int argc, char **) {
    // Reference the module so the linker keeps the dependency; never call it.
    return argc > 100 && PyInit_test_cmake_build() != nullptr ? 1 : 0;
}
