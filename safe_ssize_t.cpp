#include <cstdint>
#include <iostream>

template <typename IntType>
struct safe_ssize_t {
    IntType val;
    safe_ssize_t(const IntType &val) : val{val} {
        static_assert(sizeof(IntType) <= sizeof(std::int64_t), "");
    }
    std::int64_t as_ssize_t() const { return (std::int64_t) val; }
};

template <typename IntType>
void show(const IntType &val) {
    std::cout << safe_ssize_t<IntType>(val).as_ssize_t() << std::endl;
}

int main() {
    std::int64_t sval = INT64_MAX;
    show(sval);
    sval--;
    show(sval);
    sval = INT64_MIN;
    sval++;
    show(sval);
    sval--;
    show(sval);

    std::uint64_t uval = INT64_MAX;
    show(uval);
    uval++;
    show(uval);
    uval = UINT64_MAX;
    uval--;
    show(uval);
    uval++;
    show(uval);

    return 0;
}
