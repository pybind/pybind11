#include <pybind11/pybind11.h>
#include <pybind11/eval.h>

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

PyObject *make_embedded_module() {
    py::module m("widget_module");

    py::class_<Widget, PyWidget>(m, "Widget")
        .def(py::init<std::string>())
        .def_property_readonly("the_message", &Widget::the_message);

    return m.ptr();
}

py::object import_file(const std::string &module, const std::string &path, py::object globals) {
    auto locals = py::dict("module_name"_a=module, "path"_a=path);
    py::eval<py::eval_statements>(
        "import imp\n"
        "with open(path) as file:\n"
        "    new_module = imp.load_module(module_name, file, path, ('py', 'U', imp.PY_SOURCE))",
        globals, locals
    );
    return locals["new_module"];
}

TEST_CASE("Pass classes and data between modules defined in C++ and Python") {
    PyImport_AppendInittab("widget_module", &make_embedded_module);
    Py_Initialize();
    {
        auto globals = py::module::import("__main__").attr("__dict__");
        auto module = import_file("widget", "test_interpreter.py", globals);
        REQUIRE(py::hasattr(module, "DerivedWidget"));

        auto py_widget = module.attr("DerivedWidget")("Hello, World!");
        auto message = py_widget.attr("the_message");
        REQUIRE(message.cast<std::string>() == "Hello, World!");

        const auto &cpp_widget = py_widget.cast<const Widget &>();
        REQUIRE(cpp_widget.the_answer() == 42);
    }
    Py_Finalize();
}
