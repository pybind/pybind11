#include <pybind11/native_enum.h>

#include "pybind11_tests.h"

namespace test_native_enum {

// https://en.cppreference.com/w/cpp/language/enum

// enum that takes 16 bits
enum smallenum : std::int16_t { a, b, c };

// color may be red (value 0), yellow (value 1), green (value 20), or blue (value 21)
enum color { red, yellow, green = 20, blue };

// altitude may be altitude::high or altitude::low
enum class altitude : char {
    high = 'h',
    low = 'l', // trailing comma only allowed after CWG518
};

// the constant d is 0, the constant e is 1, the constant f is 3
enum { d, e, f = e + 2 };

// https://github.com/protocolbuffers/protobuf/blob/d70b5c5156858132decfdbae0a1103e6a5cb1345/src/google/protobuf/generated_enum_util.h#L52-L53
template <typename T>
struct is_proto_enum : std::false_type {};

enum some_proto_enum : int { Zero, One };

template <>
struct is_proto_enum<some_proto_enum> : std::true_type {};

} // namespace test_native_enum

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

template <typename ProtoEnumType>
struct type_caster_enum_type_enabled<
    ProtoEnumType,
    detail::enable_if_t<test_native_enum::is_proto_enum<ProtoEnumType>::value>> : std::false_type {
};

// https://github.com/pybind/pybind11_protobuf/blob/a50899c2eb604fc5f25deeb8901eff6231b8b3c0/pybind11_protobuf/enum_type_caster.h#L101-L105
template <typename ProtoEnumType>
struct type_caster<ProtoEnumType,
                   detail::enable_if_t<test_native_enum::is_proto_enum<ProtoEnumType>::value>> {
    static handle
    cast(const ProtoEnumType & /*src*/, return_value_policy /*policy*/, handle /*parent*/) {
        return py::none();
    }

    bool load(handle /*src*/, bool /*convert*/) {
        value = static_cast<ProtoEnumType>(0);
        return true;
    }

    PYBIND11_TYPE_CASTER(ProtoEnumType, const_name<ProtoEnumType>());
};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

TEST_SUBMODULE(native_enum, m) {
    using namespace test_native_enum;

    py::native_enum<color>(m, "color")
        .value("red", color::red)
        .value("yellow", color::yellow)
        .value("green", color::green)
        .value("blue", color::blue);

    m.def("pass_color", [](color e) { return static_cast<int>(e); });
    m.def("return_color", [](int i) { return static_cast<color>(i); });

    m.def("pass_some_proto_enum", [](some_proto_enum) { return py::none(); });
    m.def("return_some_proto_enum", []() { return some_proto_enum::Zero; });
}
