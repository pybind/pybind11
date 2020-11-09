#include <cstdio>
#include <utility>
#include <typeinfo>
#include <type_traits>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct CopyMove {
    CopyMove() = default;
    CopyMove(const CopyMove& other) { printf("copy "); }
    CopyMove(CopyMove&& other) { printf("move "); }
};

struct MoveOnly {
    MoveOnly() = default;
    MoveOnly(const MoveOnly& other) = delete;
    MoveOnly(MoveOnly&& other) { printf("move "); }
};

struct CopyOnly {
    CopyOnly() = default;
    CopyOnly(const CopyOnly& other) { printf("copy "); }
    CopyOnly(CopyOnly&& other) = delete;
};

struct MoveOnly_move {
    MoveOnly_move() = default;
    MoveOnly_move(const MoveOnly_move& other) = delete;
    MoveOnly_move(MoveOnly_move&& other) = delete;
};

template <typename T>
struct caster {
    T value;
    caster() { printf("\ncaster<%s>:\n", pybind11::type_id<T>().c_str()); }
    operator T*() { return &value; }
    operator T&() { return value; }

    template <typename T_ = T, std::enable_if_t<std::is_move_constructible<T_>::value && !std::is_abstract<T_>::value, int> = 0>
    operator T_() { return std::move(value); }
};

/// Helper template to strip away type modifiers
template <typename T> struct intrinsic_type                       { using type = T; };
template <typename T> struct intrinsic_type<const T>              { using type = typename intrinsic_type<T>::type; };
template <typename T> struct intrinsic_type<T*>                   { using type = typename intrinsic_type<T>::type; };
template <typename T> struct intrinsic_type<T&>                   { using type = typename intrinsic_type<T>::type; };
template <typename T> struct intrinsic_type<T&&>                  { using type = typename intrinsic_type<T>::type; };
template <typename T, size_t N> struct intrinsic_type<const T[N]> { using type = typename intrinsic_type<T>::type; };
template <typename T, size_t N> struct intrinsic_type<T[N]>       { using type = typename intrinsic_type<T>::type; };
template <typename T> using intrinsic_t = typename intrinsic_type<T>::type;

template <typename T>
using cast_op_type =
    py::detail::conditional_t<std::is_pointer<typename std::remove_reference<T>::type>::value,
        typename std::add_pointer<intrinsic_t<T>>::type,
        py::detail::conditional_t<std::is_rvalue_reference<T>::value,
            intrinsic_t<T>,
            typename std::add_lvalue_reference<intrinsic_t<T>>::type>>;

static_assert(std::is_same<cast_op_type<CopyMove&&>, CopyMove>::value, "rvalue reference will be moved to new T");
static_assert(std::is_same<cast_op_type<CopyMove&>, CopyMove&>::value, "lvalue reference");
static_assert(std::is_same<cast_op_type<CopyMove>, CopyMove&>::value, "lvalue reference");
static_assert(std::is_same<cast_op_type<CopyOnly&&>, CopyOnly>::value, "rvalue reference will be moved to new T");

template <typename T>
cast_op_type<T> cast_op(caster<intrinsic_t<T>> &caster) {
    return caster.operator cast_op_type<T>();
}

template <typename ARG>
void consume(ARG o) { printf(": %s(%s)\n", __FUNCTION__, pybind11::full_type_id<ARG>().c_str()); }

#define TEST(CASTER, FUNC_ARG) consume<FUNC_ARG>(cast_op<FUNC_ARG>(CASTER))

int main(int argc, char const *argv[])
{
    CopyMove x;
    consume(x);
    consume(std::move(x));

    MoveOnly y; printf("\n");
    // consume(y);  // cannot copy
    consume(std::move(y));

    CopyOnly z; printf("\n");
    consume(z);
    // consume(std::move(z));  // cannot move

    std::pair<CopyMove, MoveOnly> p1; printf("\n");
    // consume(p1);  // cannot copy
    consume(std::move(p1));  // moves

    std::pair<CopyMove, CopyOnly> p2; printf("\n");
    consume(p2);  // copies
    consume(std::move(p2));  // copies

#if 0  // can neither move nor copy
    std::pair<MoveOnly, CopyOnly> p3; printf("\n");
    consume(p3);
    consume(std::move(p3));
#endif

    caster<CopyMove> cm;
    TEST(cm, CopyMove);
    TEST(cm, CopyMove&&);
    TEST(cm, CopyMove&);

    caster<MoveOnly> nc;
    // TEST(nc, MoveOnly);  // cannot copy
    TEST(nc, MoveOnly&&);
    TEST(nc, MoveOnly&);

    caster<CopyOnly> nm;
    TEST(nm, CopyOnly);
    // TEST(nm, CopyOnly&&);  // cannot move
    TEST(nm, CopyOnly&);

    return 0;
}
