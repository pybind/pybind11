#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

namespace py = pybind11;

#ifdef PYBIND11_HAS_NATIVE_ENUM
#    include <pybind11/native_enum.h>
#endif

namespace pybind11_tests {
namespace mod_per_interpreter_gil_with_singleton {
// A singleton class that holds references to certain Python objects
// This singleton is per-interpreter using gil_safe_call_once_and_store
class MySingleton {
public:
    MySingleton() = default;
    ~MySingleton() = default;
    MySingleton(const MySingleton &) = delete;
    MySingleton &operator=(const MySingleton &) = delete;
    MySingleton(MySingleton &&) = default;
    MySingleton &operator=(MySingleton &&) = default;

    static MySingleton &get_instance() {
        PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<MySingleton> storage;
        return storage
            .call_once_and_store_result([]() -> MySingleton {
                MySingleton instance{};

                auto emplace = [&instance](const py::handle &obj) -> void {
                    obj.inc_ref(); // Ensure the object is not GC'd while interpreter is alive
                    instance.objects.emplace_back(obj);
                };

                // Example objects to store in the singleton
                emplace(py::type::handle_of(py::none()));                        // static type
                emplace(py::type::handle_of(py::tuple()));                       // static type
                emplace(py::type::handle_of(py::list()));                        // static type
                emplace(py::type::handle_of(py::dict()));                        // static type
                emplace(py::module_::import("collections").attr("OrderedDict")); // static type
                emplace(py::module_::import("collections").attr("defaultdict")); // heap type
                emplace(py::module_::import("collections").attr("deque"));       // heap type

                assert(instance.objects.size() == 7);
                return instance;
            })
            .get_stored();
    }

    std::vector<py::handle> &get_objects() { return objects; }

    static void init() {
        // Ensure the singleton is created
        auto &instance = get_instance();
        (void) instance; // suppress unused variable warning
        assert(instance.objects.size() == 7);
        // Register cleanup at interpreter exit
        py::module_::import("atexit").attr("register")(py::cpp_function(&MySingleton::clear));
    }

    static void clear() {
        auto &instance = get_instance();
        (void) instance; // suppress unused variable warning
        assert(instance.objects.size() == 7);
        for (const auto &obj : instance.objects) {
            obj.dec_ref();
        }
        instance.objects.clear();
    }

private:
    std::vector<py::handle> objects;
};

class MyClass {
public:
    explicit MyClass(py::ssize_t v) : value(v) {}
    py::ssize_t get_value() const { return value; }

private:
    py::ssize_t value;
};

class MyGlobalError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class MyLocalError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class MyEnum : int {
    ONE = 1,
    TWO = 2,
    THREE = 3,
};
} // namespace mod_per_interpreter_gil_with_singleton
} // namespace pybind11_tests

PYBIND11_MODULE(mod_per_interpreter_gil_with_singleton,
                m,
                py::mod_gil_not_used(),
                py::multiple_interpreters::per_interpreter_gil()) {
    using namespace pybind11_tests::mod_per_interpreter_gil_with_singleton;

#ifdef PYBIND11_HAS_SUBINTERPRETER_SUPPORT
    m.attr("defined_PYBIND11_HAS_SUBINTERPRETER_SUPPORT") = true;
#else
    m.attr("defined_PYBIND11_HAS_SUBINTERPRETER_SUPPORT") = false;
#endif

    MySingleton::init();

    // Ensure py::multiple_interpreters::per_interpreter_gil() works with singletons using
    // py::gil_safe_call_once_and_store
    m.def(
        "get_objects_in_singleton",
        []() -> std::vector<py::handle> { return MySingleton::get_instance().get_objects(); },
        "Get the list of objects stored in the singleton");

    // Ensure py::multiple_interpreters::per_interpreter_gil() works with class bindings
    py::class_<MyClass>(m, "MyClass")
        .def(py::init<py::ssize_t>())
        .def("get_value", &MyClass::get_value);

    // Ensure py::multiple_interpreters::per_interpreter_gil() works with global exceptions
    py::register_exception<MyGlobalError>(m, "MyGlobalError");
    // Ensure py::multiple_interpreters::per_interpreter_gil() works with local exceptions
    py::register_local_exception<MyLocalError>(m, "MyLocalError");

#ifdef PYBIND11_HAS_NATIVE_ENUM
    // Ensure py::multiple_interpreters::per_interpreter_gil() works with native_enum
    py::native_enum<MyEnum>(m, "MyEnum", "enum.IntEnum")
        .value("ONE", MyEnum::ONE)
        .value("TWO", MyEnum::TWO)
        .value("THREE", MyEnum::THREE)
        .finalize();
#else
    py::enum_<MyEnum>(m, "MyEnum")
        .value("ONE", MyEnum::ONE)
        .value("TWO", MyEnum::TWO)
        .value("THREE", MyEnum::THREE);
#endif
}
