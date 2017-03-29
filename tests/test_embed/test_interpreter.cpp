#include <pybind11/embed.h>
#include <catch.hpp>

namespace py = pybind11;
using namespace py::literals;

class Widget {
public:
    Widget(std::string message) : message(message) { }
    virtual ~Widget() = default;

    std::string the_message() const { return message; }
    virtual int the_answer() const = 0;

private:
    std::string message;
};

class PyWidget final : public Widget {
    using Widget::Widget;

    int the_answer() const override { PYBIND11_OVERLOAD_PURE(int, Widget, the_answer); }
};

PYBIND11_EMBEDDED_MODULE(widget_module, m) {
    py::class_<Widget, PyWidget>(m, "Widget")
        .def(py::init<std::string>())
        .def_property_readonly("the_message", &Widget::the_message);
}

PYBIND11_EMBEDDED_MODULE(throw_exception, ) {
    throw std::runtime_error("C++ Error");
}

PYBIND11_EMBEDDED_MODULE(throw_error_already_set, ) {
    auto d = py::dict();
    d["missing"].cast<py::object>();
}

TEST_CASE("Pass classes and data between modules defined in C++ and Python") {
    auto module = py::module::import("test_interpreter");
    REQUIRE(py::hasattr(module, "DerivedWidget"));

    auto locals = py::dict("hello"_a="Hello, World!", "x"_a=5, **module.attr("__dict__"));
    py::exec(R"(
        widget = DerivedWidget("{} - {}".format(hello, x))
        message = widget.the_message
    )", py::globals(), locals);
    REQUIRE(locals["message"].cast<std::string>() == "Hello, World! - 5");

    auto py_widget = module.attr("DerivedWidget")("The question");
    auto message = py_widget.attr("the_message");
    REQUIRE(message.cast<std::string>() == "The question");

    const auto &cpp_widget = py_widget.cast<const Widget &>();
    REQUIRE(cpp_widget.the_answer() == 42);
}

TEST_CASE("Import error handling") {
    REQUIRE_NOTHROW(py::module::import("widget_module"));
    REQUIRE_THROWS_WITH(py::module::import("throw_exception"),
                        "ImportError: C++ Error");
    REQUIRE_THROWS_WITH(py::module::import("throw_error_already_set"),
                        Catch::Contains("ImportError: KeyError"));
}

TEST_CASE("There can be only one interpreter") {
    static_assert(std::is_move_constructible<py::scoped_interpreter>::value, "");
    static_assert(!std::is_move_assignable<py::scoped_interpreter>::value, "");
    static_assert(!std::is_copy_constructible<py::scoped_interpreter>::value, "");
    static_assert(!std::is_copy_assignable<py::scoped_interpreter>::value, "");

    REQUIRE_THROWS_WITH(py::initialize_interpreter(), "The interpreter is already running");
    REQUIRE_THROWS_WITH(py::scoped_interpreter(), "The interpreter is already running");

    py::finalize_interpreter();
    REQUIRE_NOTHROW(py::scoped_interpreter());
    {
        auto pyi1 = py::scoped_interpreter();
        auto pyi2 = std::move(pyi1);
    }
    py::initialize_interpreter();
}
