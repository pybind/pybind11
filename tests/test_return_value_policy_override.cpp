#include <pybind11/smart_holder.h>

#include "pybind11_tests.h"

#include <vector>

namespace test_return_value_policy_override {

struct some_type {};

struct data_field {
    int value = -99;

    explicit data_field(int v) : value(v) {}
};

struct data_fields_holder {
    std::vector<data_field> vec;

    explicit data_fields_holder(std::size_t vec_size) {
        for (std::size_t i = 0; i < vec_size; i++) {
            vec.emplace_back(13 + static_cast<int>(i) * 11);
        }
    }

    data_field *vec_at(std::size_t index) {
        if (index >= vec.size()) {
            return nullptr;
        }
        return &vec[index];
    }

    const data_field *vec_at_const_ptr(std::size_t index) { return vec_at(index); }
};

// cp = copyable, mv = movable, 1 = yes, 0 = no
struct type_cp1_mv1 {
    std::string mtxt;
    explicit type_cp1_mv1(const std::string &mtxt_) : mtxt(mtxt_) {}
    type_cp1_mv1(const type_cp1_mv1 &other) { mtxt = other.mtxt + "_CpCtor"; }
    type_cp1_mv1(type_cp1_mv1 &&other) noexcept { mtxt = other.mtxt + "_MvCtor"; }
    type_cp1_mv1 &operator=(const type_cp1_mv1 &other) {
        mtxt = other.mtxt + "_CpCtor";
        return *this;
    }
    type_cp1_mv1 &operator=(type_cp1_mv1 &&other) noexcept {
        mtxt = other.mtxt + "_MvCtor";
        return *this;
    }
};

// nocopy
struct type_cp0_mv1 {
    std::string mtxt;
    explicit type_cp0_mv1(const std::string &mtxt_) : mtxt(mtxt_) {}
    type_cp0_mv1(const type_cp0_mv1 &) = delete;
    type_cp0_mv1(type_cp0_mv1 &&other) noexcept { mtxt = other.mtxt + "_MvCtor"; }
    type_cp0_mv1 &operator=(const type_cp0_mv1 &) = delete;
    type_cp0_mv1 &operator=(type_cp0_mv1 &&other) noexcept {
        mtxt = other.mtxt + "_MvCtor";
        return *this;
    }
};

// nomove
struct type_cp1_mv0 {
    std::string mtxt;
    explicit type_cp1_mv0(const std::string &mtxt_) : mtxt(mtxt_) {}
    type_cp1_mv0(const type_cp1_mv0 &other) { mtxt = other.mtxt + "_CpCtor"; }
    type_cp1_mv0(type_cp1_mv0 &&other) = delete;
    type_cp1_mv0 &operator=(const type_cp1_mv0 &other) {
        mtxt = other.mtxt + "_CpCtor";
        return *this;
    }
    type_cp1_mv0 &operator=(type_cp1_mv0 &&other) = delete;
};

// nocopy_nomove
struct type_cp0_mv0 {
    std::string mtxt;
    explicit type_cp0_mv0(const std::string &mtxt_) : mtxt(mtxt_) {}
    type_cp0_mv0(const type_cp0_mv0 &) = delete;
    type_cp0_mv0(type_cp0_mv0 &&other) = delete;
    type_cp0_mv0 &operator=(const type_cp0_mv0 &other) = delete;
    type_cp0_mv0 &operator=(type_cp0_mv0 &&other) = delete;
};

type_cp1_mv1 return_value() { return type_cp1_mv1{"value"}; }

type_cp1_mv1 *return_pointer() {
    static type_cp1_mv1 value("pointer");
    return &value;
}

const type_cp1_mv1 *return_const_pointer() {
    static type_cp1_mv1 value("const_pointer");
    return &value;
}

type_cp1_mv1 &return_reference() {
    static type_cp1_mv1 value("reference");
    return value;
}

const type_cp1_mv1 &return_const_reference() {
    static type_cp1_mv1 value("const_reference");
    return value;
}

std::shared_ptr<type_cp1_mv1> return_shared_pointer() {
    return std::make_shared<type_cp1_mv1>("shared_pointer");
}

std::unique_ptr<type_cp1_mv1> return_unique_pointer() {
    return std::unique_ptr<type_cp1_mv1>(new type_cp1_mv1("unique_pointer"));
}

type_cp0_mv1 return_value_nocopy() { return type_cp0_mv1{"value_nocopy"}; }

type_cp0_mv1 *return_pointer_nocopy() {
    static type_cp0_mv1 value("pointer_nocopy");
    return &value;
}

const type_cp0_mv1 *return_const_pointer_nocopy() {
    static type_cp0_mv1 value("const_pointer_nocopy");
    return &value;
}

type_cp0_mv1 &return_reference_nocopy() {
    static type_cp0_mv1 value("reference_nocopy");
    return value;
}

std::shared_ptr<type_cp0_mv1> return_shared_pointer_nocopy() {
    return std::make_shared<type_cp0_mv1>("shared_pointer_nocopy");
}

std::unique_ptr<type_cp0_mv1> return_unique_pointer_nocopy() {
    return std::unique_ptr<type_cp0_mv1>(new type_cp0_mv1("unique_pointer_nocopy"));
}

type_cp1_mv0 *return_pointer_nomove() {
    static type_cp1_mv0 value("pointer_nomove");
    return &value;
}

const type_cp1_mv0 *return_const_pointer_nomove() {
    static type_cp1_mv0 value("const_pointer_nomove");
    return &value;
}

type_cp1_mv0 &return_reference_nomove() {
    static type_cp1_mv0 value("reference_nomove");
    return value;
}

const type_cp1_mv0 &return_const_reference_nomove() {
    static type_cp1_mv0 value("const_reference_nomove");
    return value;
}

std::shared_ptr<type_cp1_mv0> return_shared_pointer_nomove() {
    return std::make_shared<type_cp1_mv0>("shared_pointer_nomove");
}

std::unique_ptr<type_cp1_mv0> return_unique_pointer_nomove() {
    return std::unique_ptr<type_cp1_mv0>(new type_cp1_mv0("unique_pointer_nomove"));
}

type_cp0_mv0 *return_pointer_nocopy_nomove() {
    static type_cp0_mv0 value("pointer_nocopy_nomove");
    return &value;
}

std::shared_ptr<type_cp0_mv0> return_shared_pointer_nocopy_nomove() {
    return std::make_shared<type_cp0_mv0>("shared_pointer_nocopy_nomove");
}

std::unique_ptr<type_cp0_mv0> return_unique_pointer_nocopy_nomove() {
    return std::unique_ptr<type_cp0_mv0>(new type_cp0_mv0("unique_pointer_nocopy_nomove"));
}

} // namespace test_return_value_policy_override

using test_return_value_policy_override::data_field;
using test_return_value_policy_override::data_fields_holder;
using test_return_value_policy_override::some_type;
using test_return_value_policy_override::type_cp0_mv0;
using test_return_value_policy_override::type_cp0_mv1;
using test_return_value_policy_override::type_cp1_mv0;
using test_return_value_policy_override::type_cp1_mv1;

namespace pybind11 {
namespace detail {

const char *return_value_policy_name(return_value_policy policy) {
    switch (policy) {
        case return_value_policy::automatic:
            return "automatic";
        case return_value_policy::automatic_reference:
            return "automatic_reference";
        case return_value_policy::take_ownership:
            return "take_ownership";
        case return_value_policy::copy:
            return "copy";
        case return_value_policy::move:
            return "move";
        case return_value_policy::reference:
            return "reference";
        case return_value_policy::reference_internal:
            return "reference_internal";
        case return_value_policy::_return_as_bytes:
            return "_return_as_bytes";
        case return_value_policy::_clif_automatic:
            return "_clif_automatic";
        default:
            return "Expected to be unreachable.";
    }
};

template <>
struct type_caster<some_type> : type_caster_base<some_type> {

    static handle cast(some_type &&, return_value_policy policy, handle /*parent*/) {
        return str(std::string(return_value_policy_name(policy))).release().ptr();
    }

    static handle cast(some_type *, return_value_policy policy, handle /*parent*/) {
        return str(std::string(return_value_policy_name(policy))).release().ptr();
    }
};

} // namespace detail
} // namespace pybind11

PYBIND11_SMART_HOLDER_TYPE_CASTERS(data_field)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(data_fields_holder)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(type_cp1_mv1)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(type_cp0_mv1)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(type_cp1_mv0)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(type_cp0_mv0)

TEST_SUBMODULE(return_value_policy_override, m) {
    m.def("return_value_with_default_policy", []() { return some_type(); });
    m.def(
        "return_value_with_policy_copy",
        []() { return some_type(); },
        py::return_value_policy::copy);
    m.def(
        "return_value_with_policy_clif_automatic",
        []() { return some_type(); },
        py::return_value_policy::_clif_automatic);
    m.def("return_pointer_with_default_policy", []() {
        static some_type value;
        return &value;
    });
    m.def(
        "return_pointer_with_policy_move",
        []() {
            static some_type value;
            return &value;
        },
        py::return_value_policy::move);
    m.def(
        "return_pointer_with_policy_clif_automatic",
        []() {
            static some_type value;
            return &value;
        },
        py::return_value_policy::_clif_automatic);

    py::classh<data_field>(m, "data_field").def_readwrite("value", &data_field::value);
    py::classh<data_fields_holder>(m, "data_fields_holder")
        .def(py::init<std::size_t>())
        .def("vec_at",
             [](const py::object &self_py, std::size_t index) {
                 auto *self = py::cast<data_fields_holder *>(self_py);
                 return py::cast(
                     self->vec_at(index), py::return_value_policy::_clif_automatic, self_py);
             })
        .def("vec_at_const_ptr", [](const py::object &self_py, std::size_t index) {
            auto *self = py::cast<data_fields_holder *>(self_py);
            return py::cast(
                self->vec_at_const_ptr(index), py::return_value_policy::_clif_automatic, self_py);
        });

    py::classh<type_cp1_mv1>(m, "type_cp1_mv1")
        .def(py::init<std::string>())
        .def_readonly("mtxt", &type_cp1_mv1::mtxt);
    m.def("return_value",
          &test_return_value_policy_override::return_value,
          py::return_value_policy::_clif_automatic);
    m.def("return_pointer",
          &test_return_value_policy_override::return_pointer,
          py::return_value_policy::_clif_automatic);
    m.def("return_const_pointer",
          &test_return_value_policy_override::return_const_pointer,
          py::return_value_policy::_clif_automatic);
    m.def("return_reference",
          &test_return_value_policy_override::return_reference,
          py::return_value_policy::_clif_automatic);
    m.def("return_const_reference",
          &test_return_value_policy_override::return_const_reference,
          py::return_value_policy::_clif_automatic);
    m.def("return_unique_pointer",
          &test_return_value_policy_override::return_unique_pointer,
          py::return_value_policy::_clif_automatic);
    m.def("return_shared_pointer",
          &test_return_value_policy_override::return_shared_pointer,
          py::return_value_policy::_clif_automatic);

    py::classh<type_cp0_mv1>(m, "type_cp0_mv1")
        .def(py::init<std::string>())
        .def_readonly("mtxt", &type_cp0_mv1::mtxt);
    m.def("return_value_nocopy",
          &test_return_value_policy_override::return_value_nocopy,
          py::return_value_policy::_clif_automatic);
    m.def("return_pointer_nocopy",
          &test_return_value_policy_override::return_pointer_nocopy,
          py::return_value_policy::_clif_automatic);
    m.def("return_const_pointer_nocopy",
          &test_return_value_policy_override::return_const_pointer_nocopy,
          py::return_value_policy::_clif_automatic);
    m.def("return_reference_nocopy",
          &test_return_value_policy_override::return_reference_nocopy,
          py::return_value_policy::_clif_automatic);
    m.def("return_shared_pointer_nocopy",
          &test_return_value_policy_override::return_shared_pointer_nocopy,
          py::return_value_policy::_clif_automatic);
    m.def("return_unique_pointer_nocopy",
          &test_return_value_policy_override::return_unique_pointer_nocopy,
          py::return_value_policy::_clif_automatic);

    py::classh<type_cp1_mv0>(m, "type_cp1_mv0")
        .def(py::init<std::string>())
        .def_readonly("mtxt", &type_cp1_mv0::mtxt);
    m.def("return_pointer_nomove",
          &test_return_value_policy_override::return_pointer_nomove,
          py::return_value_policy::_clif_automatic);
    m.def("return_const_pointer_nomove",
          &test_return_value_policy_override::return_const_pointer_nomove,
          py::return_value_policy::_clif_automatic);
    m.def("return_reference_nomove",
          &test_return_value_policy_override::return_reference_nomove,
          py::return_value_policy::_clif_automatic);
    m.def("return_const_reference_nomove",
          &test_return_value_policy_override::return_const_reference_nomove,
          py::return_value_policy::_clif_automatic);
    m.def("return_shared_pointer_nomove",
          &test_return_value_policy_override::return_shared_pointer_nomove,
          py::return_value_policy::_clif_automatic);
    m.def("return_unique_pointer_nomove",
          &test_return_value_policy_override::return_unique_pointer_nomove,
          py::return_value_policy::_clif_automatic);

    py::classh<type_cp0_mv0>(m, "type_cp0_mv0")
        .def(py::init<std::string>())
        .def_readonly("mtxt", &type_cp0_mv0::mtxt);
    m.def("return_pointer_nocopy_nomove",
          &test_return_value_policy_override::return_pointer_nocopy_nomove,
          py::return_value_policy::_clif_automatic);
    m.def("return_shared_pointer_nocopy_nomove",
          &test_return_value_policy_override::return_shared_pointer_nocopy_nomove,
          py::return_value_policy::_clif_automatic);
    m.def("return_unique_pointer_nocopy_nomove",
          &test_return_value_policy_override::return_unique_pointer_nocopy_nomove,
          py::return_value_policy::_clif_automatic);
}
