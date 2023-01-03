import timeit

import custom

print(custom.test_me(5, 2))

import raw_custom

print(raw_custom.test_me(5, 2))

for _ in range(4):
    print(
        "pybind11",
        timeit.timeit("custom.test_me(5, 2)", "import custom", number=100000000),
    )
    print(
        "raw",
        timeit.timeit(
            "raw_custom.test_me(5, 2)", "import raw_custom", number=100000000
        ),
    )
