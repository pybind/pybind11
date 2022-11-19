#include <pybind11/native_enum.h>

#include "pybind11_tests.h"

#include <typeindex>

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

enum class export_values { exv0, exv1 };

enum class member_doc { mem0, mem1, mem2 };

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

    m.attr("native_enum_type_map_abi_id_c_str")
        = py::cross_extension_shared_states::native_enum_type_map::abi_id();

    m += py::native_enum<smallenum>("smallenum")
             .value("a", smallenum::a)
             .value("b", smallenum::b)
             .value("c", smallenum::c);

    m += py::native_enum<color>("color")
             .value("red", color::red)
             .value("yellow", color::yellow)
             .value("green", color::green)
             .value("blue", color::blue);

    m += py::native_enum<altitude>("altitude")
             .value("high", altitude::high)
             .value("low", altitude::low);

    m += py::native_enum<export_values>("export_values")
             .value("exv0", export_values::exv0)
             .value("exv1", export_values::exv1)
             .export_values();

    m += py::native_enum<member_doc>("member_doc")
             .value("mem0", member_doc::mem0, "docA")
             .value("mem1", member_doc::mem1)
             .value("mem2", member_doc::mem2, "docC");

    m.def("isinstance_color", [](const py::object &obj) { return py::isinstance<color>(obj); });

    m.def("pass_color", [](color e) { return static_cast<int>(e); });
    m.def("return_color", [](int i) { return static_cast<color>(i); });

    m.def("pass_some_proto_enum", [](some_proto_enum) { return py::none(); });
    m.def("return_some_proto_enum", []() { return some_proto_enum::Zero; });

#if defined(__MINGW32__)
    m.attr("obj_cast_color_ptr") = "MinGW: dangling pointer to an unnamed temporary may be used "
                                   "[-Werror=dangling-pointer=]";
#elif defined(NDEBUG)
    m.attr("obj_cast_color_ptr") = "NDEBUG disables cast safety check";
#else
    m.def("obj_cast_color_ptr", [](const py::object &obj) { obj.cast<color *>(); });
#endif

    m.def("native_enum_data_was_not_added_error_message", [](const char *enum_name) {
        py::detail::native_enum_data data(enum_name, std::type_index(typeid(void)), false);
        data.disarm_correct_use_check();
        return data.was_not_added_error_message();
    });

    m.def("native_enum_ctor_malformed_utf8", [](const char *malformed_utf8) {
        enum fake { x };
        py::native_enum<fake>{malformed_utf8};
    });

    m.def("native_enum_value_malformed_utf8", [](const char *malformed_utf8) {
        enum fake { x };
        py::native_enum<fake>("fake").value(malformed_utf8, fake::x);
    });

    m.def("double_registration_native_enum", [](py::module_ m) {
        enum fake { x };
        m += py::native_enum<fake>("fake_double_registration_native_enum").value("x", fake::x);
        py::native_enum<fake>("fake_double_registration_native_enum");
    });

    m.def("native_enum_name_clash", [](py::module_ m) {
        enum fake { x };
        m += py::native_enum<fake>("fake_native_enum_name_clash").value("x", fake::x);
    });

    m.def("native_enum_value_name_clash", [](py::module_ m) {
        enum fake { x };
        m += py::native_enum<fake>("fake_native_enum_value_name_clash")
                 .value("fake_native_enum_value_name_clash_x", fake::x)
                 .export_values();
    });

    m.def("double_registration_enum_before_native_enum", [](const py::module_ &m) {
        enum fake { x };
        py::enum_<fake>(m, "fake_enum_first").value("x", fake::x);
        py::native_enum<fake>("fake_enum_first").value("x", fake::x);
    });

    m.def("double_registration_native_enum_before_enum", [](py::module_ m) {
        enum fake { x };
        m += py::native_enum<fake>("fake_native_enum_first").value("x", fake::x);
        py::enum_<fake>(m, "name_must_be_different_to_reach_desired_code_path");
    });

#if defined(PYBIND11_NEGATE_THIS_CONDITION_FOR_LOCAL_TESTING) && !defined(NDEBUG)
    m.def("native_enum_correct_use_failure", []() {
        enum fake { x };
        py::native_enum<fake>("fake_native_enum_correct_use_failure").value("x", fake::x);
    });
#else
    m.attr("native_enum_correct_use_failure") = "For local testing only: terminates process";
#endif
}
