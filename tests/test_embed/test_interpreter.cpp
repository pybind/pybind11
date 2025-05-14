#include <pybind11/embed.h>

// Silence MSVC C++17 deprecation warning from Catch regarding std::uncaught_exceptions (up to
// catch 2.0.1; this should be fixed in the next catch release after 2.0.1).
PYBIND11_WARNING_DISABLE_MSVC(4996)

#include <catch.hpp>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <thread>
#include <utility>

namespace py = pybind11;
using namespace py::literals;

size_t get_sys_path_size() {
    auto sys_path = py::module::import("sys").attr("path");
    return py::len(sys_path);
}

class Widget {
public:
    explicit Widget(std::string message) : message(std::move(message)) {}
    virtual ~Widget() = default;

    std::string the_message() const { return message; }
    virtual int the_answer() const = 0;
    virtual std::string argv0() const = 0;

private:
    std::string message;
};

class PyWidget final : public Widget {
    using Widget::Widget;

    int the_answer() const override { PYBIND11_OVERRIDE_PURE(int, Widget, the_answer); }
    std::string argv0() const override { PYBIND11_OVERRIDE_PURE(std::string, Widget, argv0); }
};

class test_override_cache_helper {

public:
    virtual int func() { return 0; }

    test_override_cache_helper() = default;
    virtual ~test_override_cache_helper() = default;
    // Non-copyable
    test_override_cache_helper &operator=(test_override_cache_helper const &Right) = delete;
    test_override_cache_helper(test_override_cache_helper const &Copy) = delete;
};

class test_override_cache_helper_trampoline : public test_override_cache_helper {
    int func() override { PYBIND11_OVERRIDE(int, test_override_cache_helper, func); }
};

PYBIND11_EMBEDDED_MODULE(widget_module, m) {
    py::class_<Widget, PyWidget>(m, "Widget")
        .def(py::init<std::string>())
        .def_property_readonly("the_message", &Widget::the_message);

    m.def("add", [](int i, int j) { return i + j; });

    auto sub = m.def_submodule("sub");
    sub.def("add", [](int i, int j) { return i + j; });
}

PYBIND11_EMBEDDED_MODULE(trampoline_module, m) {
    py::class_<test_override_cache_helper,
               test_override_cache_helper_trampoline,
               std::shared_ptr<test_override_cache_helper>>(m, "test_override_cache_helper")
        .def(py::init_alias<>())
        .def("func", &test_override_cache_helper::func);
}

PYBIND11_EMBEDDED_MODULE(throw_exception, ) { throw std::runtime_error("C++ Error"); }

PYBIND11_EMBEDDED_MODULE(throw_error_already_set, ) {
    auto d = py::dict();
    d["missing"].cast<py::object>();
}

TEST_CASE("PYTHONPATH is used to update sys.path") {
    // The setup for this TEST_CASE is in catch.cpp!
    auto sys_path = py::str(py::module_::import("sys").attr("path")).cast<std::string>();
    REQUIRE_THAT(sys_path,
                 Catch::Matchers::Contains("pybind11_test_embed_PYTHONPATH_2099743835476552"));
}

TEST_CASE("Pass classes and data between modules defined in C++ and Python") {
    auto module_ = py::module_::import("test_interpreter");
    REQUIRE(py::hasattr(module_, "DerivedWidget"));

    auto locals = py::dict("hello"_a = "Hello, World!", "x"_a = 5, **module_.attr("__dict__"));
    py::exec(R"(
        widget = DerivedWidget("{} - {}".format(hello, x))
        message = widget.the_message
    )",
             py::globals(),
             locals);
    REQUIRE(locals["message"].cast<std::string>() == "Hello, World! - 5");

    auto py_widget = module_.attr("DerivedWidget")("The question");
    auto message = py_widget.attr("the_message");
    REQUIRE(message.cast<std::string>() == "The question");

    const auto &cpp_widget = py_widget.cast<const Widget &>();
    REQUIRE(cpp_widget.the_answer() == 42);
}

TEST_CASE("Override cache") {
    auto module_ = py::module_::import("test_trampoline");
    REQUIRE(py::hasattr(module_, "func"));
    REQUIRE(py::hasattr(module_, "func2"));

    auto locals = py::dict(**module_.attr("__dict__"));

    int i = 0;
    for (; i < 1500; ++i) {
        std::shared_ptr<test_override_cache_helper> p_obj;
        std::shared_ptr<test_override_cache_helper> p_obj2;

        py::object loc_inst = locals["func"]();
        p_obj = py::cast<std::shared_ptr<test_override_cache_helper>>(loc_inst);

        int ret = p_obj->func();

        REQUIRE(ret == 42);

        loc_inst = locals["func2"]();

        p_obj2 = py::cast<std::shared_ptr<test_override_cache_helper>>(loc_inst);

        p_obj2->func();
    }
}

TEST_CASE("Import error handling") {
    REQUIRE_NOTHROW(py::module_::import("widget_module"));
    REQUIRE_THROWS_WITH(py::module_::import("throw_exception"), "ImportError: C++ Error");
    REQUIRE_THROWS_WITH(py::module_::import("throw_error_already_set"),
                        Catch::Contains("ImportError: initialization failed"));

    auto locals = py::dict("is_keyerror"_a = false, "message"_a = "not set");
    py::exec(R"(
        try:
            import throw_error_already_set
        except ImportError as e:
            is_keyerror = type(e.__cause__) == KeyError
            message = str(e.__cause__)
    )",
             py::globals(),
             locals);
    REQUIRE(locals["is_keyerror"].cast<bool>() == true);
    REQUIRE(locals["message"].cast<std::string>() == "'missing'");
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

#if PY_VERSION_HEX >= PYBIND11_PYCONFIG_SUPPORT_PY_VERSION_HEX
TEST_CASE("Custom PyConfig") {
    py::finalize_interpreter();
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    REQUIRE_NOTHROW(py::scoped_interpreter{&config});
    {
        py::scoped_interpreter p{&config};
        REQUIRE(py::module_::import("widget_module").attr("add")(1, 41).cast<int>() == 42);
    }
    py::initialize_interpreter();
}

TEST_CASE("scoped_interpreter with PyConfig_InitIsolatedConfig and argv") {
    py::finalize_interpreter();
    {
        PyConfig config;
        PyConfig_InitIsolatedConfig(&config);
        char *argv[] = {strdup("a.out")};
        py::scoped_interpreter argv_scope{&config, 1, argv};
        std::free(argv[0]);
        auto module = py::module::import("test_interpreter");
        auto py_widget = module.attr("DerivedWidget")("The question");
        const auto &cpp_widget = py_widget.cast<const Widget &>();
        REQUIRE(cpp_widget.argv0() == "a.out");
    }
    py::initialize_interpreter();
}

TEST_CASE("scoped_interpreter with PyConfig_InitPythonConfig and argv") {
    py::finalize_interpreter();
    {
        PyConfig config;
        PyConfig_InitPythonConfig(&config);

        // `initialize_interpreter() overrides the default value for config.parse_argv (`1`) by
        // changing it to `0`. This test exercises `scoped_interpreter` with the default config.
        char *argv[] = {strdup("a.out"), strdup("arg1")};
        py::scoped_interpreter argv_scope(&config, 2, argv);
        std::free(argv[0]);
        std::free(argv[1]);
        auto module = py::module::import("test_interpreter");
        auto py_widget = module.attr("DerivedWidget")("The question");
        const auto &cpp_widget = py_widget.cast<const Widget &>();
        REQUIRE(cpp_widget.argv0() == "arg1");
    }
    py::initialize_interpreter();
}
#endif

TEST_CASE("Add program dir to path pre-PyConfig") {
    py::finalize_interpreter();
    size_t path_size_add_program_dir_to_path_false = 0;
    {
        py::scoped_interpreter scoped_interp{true, 0, nullptr, false};
        path_size_add_program_dir_to_path_false = get_sys_path_size();
    }
    {
        py::scoped_interpreter scoped_interp{};
        REQUIRE(get_sys_path_size() == path_size_add_program_dir_to_path_false + 1);
    }
    py::initialize_interpreter();
}

#if PY_VERSION_HEX >= PYBIND11_PYCONFIG_SUPPORT_PY_VERSION_HEX
TEST_CASE("Add program dir to path using PyConfig") {
    py::finalize_interpreter();
    size_t path_size_add_program_dir_to_path_false = 0;
    {
        PyConfig config;
        PyConfig_InitPythonConfig(&config);
        py::scoped_interpreter scoped_interp{&config, 0, nullptr, false};
        path_size_add_program_dir_to_path_false = get_sys_path_size();
    }
    {
        PyConfig config;
        PyConfig_InitPythonConfig(&config);
        py::scoped_interpreter scoped_interp{&config};
        REQUIRE(get_sys_path_size() == path_size_add_program_dir_to_path_false + 1);
    }
    py::initialize_interpreter();
}
#endif

bool has_state_dict_internals_obj() {
    py::dict state = py::detail::get_python_state_dict();
    return state.contains(PYBIND11_INTERNALS_ID);
}

bool has_pybind11_internals_static() {
    auto *&ipp = py::detail::get_internals_pp<py::detail::internals>();
    return (ipp != nullptr) && *ipp;
}

uintptr_t get_details_as_uintptr() {
    return reinterpret_cast<uintptr_t>(
        py::detail::get_internals_pp<py::detail::internals>()->get());
}

TEST_CASE("Restart the interpreter") {
    // Verify pre-restart state.
    REQUIRE(py::module_::import("widget_module").attr("add")(1, 2).cast<int>() == 3);
    REQUIRE(has_state_dict_internals_obj());
    REQUIRE(has_pybind11_internals_static());
    REQUIRE(py::module_::import("external_module").attr("A")(123).attr("value").cast<int>()
            == 123);

    // local and foreign module internals should point to the same internals:
    REQUIRE(get_details_as_uintptr()
            == py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>());

    // Restart the interpreter.
    py::finalize_interpreter();
    REQUIRE(Py_IsInitialized() == 0);

    py::initialize_interpreter();
    REQUIRE(Py_IsInitialized() == 1);

    // Internals are deleted after a restart.
    REQUIRE_FALSE(has_state_dict_internals_obj());
    REQUIRE_FALSE(has_pybind11_internals_static());
    pybind11::detail::get_internals();
    REQUIRE(has_state_dict_internals_obj());
    REQUIRE(has_pybind11_internals_static());
    REQUIRE(get_details_as_uintptr()
            == py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>());

    // Make sure that an interpreter with no get_internals() created until finalize still gets the
    // internals destroyed
    py::finalize_interpreter();
    py::initialize_interpreter();
    bool ran = false;
    py::module_::import("__main__").attr("internals_destroy_test")
        = py::capsule(&ran, [](void *ran) {
              py::detail::get_internals();
              REQUIRE(has_state_dict_internals_obj());
              REQUIRE(has_pybind11_internals_static());
              *static_cast<bool *>(ran) = true;
          });
    REQUIRE_FALSE(has_state_dict_internals_obj());
    REQUIRE_FALSE(has_pybind11_internals_static());
    REQUIRE_FALSE(ran);
    py::finalize_interpreter();
    REQUIRE(ran);
    REQUIRE_FALSE(has_pybind11_internals_static());
    py::initialize_interpreter();
    REQUIRE_FALSE(has_state_dict_internals_obj());
    REQUIRE_FALSE(has_pybind11_internals_static());

    // C++ modules can be reloaded.
    auto cpp_module = py::module_::import("widget_module");
    REQUIRE(cpp_module.attr("add")(1, 2).cast<int>() == 3);

    // Also verify submodules work
    REQUIRE(cpp_module.attr("sub").attr("add")(1, 41).cast<int>() == 42);

    // C++ type information is reloaded and can be used in python modules.
    auto py_module = py::module_::import("test_interpreter");
    auto py_widget = py_module.attr("DerivedWidget")("Hello after restart");
    REQUIRE(py_widget.attr("the_message").cast<std::string>() == "Hello after restart");
}

TEST_CASE("Subinterpreter") {
    py::module_::import("external_module"); // in the main interpreter

    // Add tags to the modules in the main interpreter and test the basics.
    py::module_::import("__main__").attr("main_tag") = "main interpreter";
    {
        auto m = py::module_::import("widget_module");
        m.attr("extension_module_tag") = "added to module in main interpreter";

        REQUIRE(m.attr("add")(1, 2).cast<int>() == 3);
    }
    REQUIRE(has_state_dict_internals_obj());
    REQUIRE(has_pybind11_internals_static());

    /// Create and switch to a subinterpreter.
    auto *main_tstate = PyThreadState_Get();
    auto *sub_tstate = Py_NewInterpreter();

    py::detail::get_num_interpreters_seen()++;

    // Subinterpreters get their own copy of builtins.
    REQUIRE_FALSE(has_state_dict_internals_obj());

#if defined(PYBIND11_SUBINTERPRETER_SUPPORT) && PY_VERSION_HEX >= 0x030C0000
    // internals hasn't been populated yet, but will be different for the subinterpreter
    REQUIRE_FALSE(has_pybind11_internals_static());

    py::list(py::module_::import("sys").attr("path")).append(py::str("."));

    auto ext_int = py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>();
    py::detail::get_internals();
    REQUIRE(has_pybind11_internals_static());
    REQUIRE(get_details_as_uintptr() == ext_int);
#else
    // This static is still defined
    REQUIRE(has_pybind11_internals_static());
#endif

    // Modules tags should be gone.
    REQUIRE_FALSE(py::hasattr(py::module_::import("__main__"), "tag"));
    {
        auto m = py::module_::import("widget_module");
        REQUIRE_FALSE(py::hasattr(m, "extension_module_tag"));

        // Function bindings should still work.
        REQUIRE(m.attr("add")(1, 2).cast<int>() == 3);
    }

    // The subinterpreter now has internals populated since we imported a pybind11 module
    REQUIRE(has_pybind11_internals_static());

    // Restore main interpreter.
    Py_EndInterpreter(sub_tstate);
    py::detail::get_num_interpreters_seen() = 1;
    PyThreadState_Swap(main_tstate);

    REQUIRE(py::hasattr(py::module_::import("__main__"), "main_tag"));
    REQUIRE(py::hasattr(py::module_::import("widget_module"), "extension_module_tag"));
    REQUIRE(has_state_dict_internals_obj());
}

#if defined(PYBIND11_SUBINTERPRETER_SUPPORT)
TEST_CASE("Multiple Subinterpreters") {
    // Make sure the module is in the main interpreter and save its pointer
    auto *main_ext = py::module_::import("external_module").ptr();
    auto main_int
        = py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>();
    py::module_::import("external_module").attr("multi_interp") = "1";

    auto *main_tstate = PyThreadState_Get();

    /// Create and switch to a subinterpreter.
    auto *sub1_tstate = Py_NewInterpreter();
    py::detail::get_num_interpreters_seen()++;

    py::list(py::module_::import("sys").attr("path")).append(py::str("."));

    // The subinterpreter has its own copy of this module which is completely separate from main
    auto *sub1_ext = py::module_::import("external_module").ptr();
    REQUIRE(sub1_ext != main_ext);
    REQUIRE_FALSE(py::hasattr(py::module_::import("external_module"), "multi_interp"));
    py::module_::import("external_module").attr("multi_interp") = "2";
    // The subinterpreter also has its own internals
    auto sub1_int
        = py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>();
    REQUIRE(sub1_int != main_int);

    // Create another interpreter
    auto *sub2_tstate = Py_NewInterpreter();
    py::detail::get_num_interpreters_seen()++;

    py::list(py::module_::import("sys").attr("path")).append(py::str("."));

    // The second subinterpreter is separate from both main and the other subinterpreter
    auto *sub2_ext = py::module_::import("external_module").ptr();
    REQUIRE(sub2_ext != main_ext);
    REQUIRE(sub2_ext != sub1_ext);
    REQUIRE_FALSE(py::hasattr(py::module_::import("external_module"), "multi_interp"));
    py::module_::import("external_module").attr("multi_interp") = "3";
    // The subinterpreter also has its own internals
    auto sub2_int
        = py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>();
    REQUIRE(sub2_int != main_int);
    REQUIRE(sub2_int != sub1_int);

    PyThreadState_Swap(sub1_tstate); // go back to sub1

    REQUIRE(py::cast<std::string>(py::module_::import("external_module").attr("multi_interp"))
            == "2");

    PyThreadState_Swap(main_tstate); // go back to main

    auto post_int
        = py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>();
    // Make sure internals went back the way it was before
    REQUIRE(main_int == post_int);

    REQUIRE(py::cast<std::string>(py::module_::import("external_module").attr("multi_interp"))
            == "1");

    PyThreadState_Swap(sub1_tstate);
    Py_EndInterpreter(sub1_tstate);
    PyThreadState_Swap(sub2_tstate);
    Py_EndInterpreter(sub2_tstate);

    py::detail::get_num_interpreters_seen() = 1;
    PyThreadState_Swap(main_tstate);
}
#endif

#if defined(Py_MOD_PER_INTERPRETER_GIL_SUPPORTED) && defined(PYBIND11_SUBINTERPRETER_SUPPORT)
TEST_CASE("Per-Subinterpreter GIL") {
    auto main_int
        = py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>();

    std::atomic<int> started, sync, failure;
    started = 0;
    sync = 0;
    failure = 0;

// REQUIRE throws on failure, so we can't use it within the thread
#    define T_REQUIRE(status)                                                                     \
        do {                                                                                      \
            assert(status);                                                                       \
            if (!(status))                                                                        \
                ++failure;                                                                        \
        } while (0)

    auto &&thread_main = [&](int num) {
        while (started == 0)
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        ++started;

        py::gil_scoped_acquire gil;
        auto main_tstate = PyThreadState_Get();

        // we have the GIL, we can access the main interpreter
        auto t_int
            = py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>();
        T_REQUIRE(t_int == main_int);
        py::module_::import("external_module").attr("multi_interp") = "1";

        PyThreadState *sub = nullptr;
        PyInterpreterConfig cfg;
        memset(&cfg, 0, sizeof(cfg));
        cfg.check_multi_interp_extensions = 1;
        cfg.gil = PyInterpreterConfig_OWN_GIL;
        auto status = Py_NewInterpreterFromConfig(&sub, &cfg);
        T_REQUIRE(!PyStatus_IsError(status));

        py::detail::get_num_interpreters_seen()++;

        py::list(py::module_::import("sys").attr("path")).append(py::str("."));

        // we have switched to the new interpreter and released the main gil

        // widget_module did not provide the mod_per_interpreter_gil tag, so it cannot be imported
        bool caught = false;
        try {
            py::module_::import("widget_module");
        } catch (pybind11::error_already_set &pe) {
            T_REQUIRE(pe.matches(PyExc_ImportError));
            std::string msg(pe.what());
            T_REQUIRE(msg.find("does not support loading in subinterpreters")
                      != std::string::npos);
            caught = true;
        }
        T_REQUIRE(caught);

        T_REQUIRE(!py::hasattr(py::module_::import("external_module"), "multi_interp"));
        py::module_::import("external_module").attr("multi_interp") = std::to_string(num);

        // wait for something to set sync to our thread number
        // we are holding our subinterpreter's GIL
        while (sync != num)
            std::this_thread::sleep_for(std::chrono::microseconds(1));

        // now change it so the next thread can mvoe on
        ++sync;

        // but keep holding the GIL until after the next thread moves on as well
        while (sync == num + 1)
            std::this_thread::sleep_for(std::chrono::microseconds(1));

        // one last check before quitting the thread, the internals should be different
        auto sub_int
            = py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>();
        T_REQUIRE(sub_int != main_int);

        Py_EndInterpreter(sub);

        PyThreadState_Swap(
            main_tstate); // switch back so the scoped_acquire can release the GIL properly
    };

    std::thread t1(thread_main, 1);
    std::thread t2(thread_main, 2);

    // we spawned two threads, at this point they are both waiting for started to increase
    ++started;

    // ok now wait for the threads to start
    while (started != 3)
        std::this_thread::sleep_for(std::chrono::microseconds(1));

    // we still hold the main GIL, at this point both threads are waiting on the main GIL
    // IN THE CASE of free threading, the threads are waiting on sync (because there is no GIL)

    // IF the below code hangs in one of the wait loops, then the child thread GIL behavior did not
    // function as expected.
    {
        // release the GIL and allow the threads to run
        py::gil_scoped_release nogil;

        // the threads are now waiting on the sync
        REQUIRE(sync == 0);

        // this will trigger thread 1 and then advance and trigger 2 and then advance
        sync = 1;

        // wait for thread 2 to advance
        while (sync != 3)
            std::this_thread::sleep_for(std::chrono::microseconds(1));

        // we know now that thread 1 has run and may be finishing
        // and thread 2 is waiting for permission to advance

        // so we move sync so that thread 2 can finish executing
        ++sync;

        // now wait for both threads to complete
        t1.join();
        t2.join();
    }

    // now we have the gil again, sanity check
    REQUIRE(py::cast<std::string>(py::module_::import("external_module").attr("multi_interp"))
            == "1");

    // the threads are stopped. we can now lower this for the rest of the test
    py::detail::get_num_interpreters_seen() = 1;

    // make sure nothing unexpected happened inside the threads, now that they are completed
    REQUIRE(failure == 0);
#    undef T_REQUIRE
}
#endif

TEST_CASE("Execution frame") {
    // When the interpreter is embedded, there is no execution frame, but `py::exec`
    // should still function by using reasonable globals: `__main__.__dict__`.
    py::exec("var = dict(number=42)");
    REQUIRE(py::globals()["var"]["number"].cast<int>() == 42);
}

TEST_CASE("Threads") {
    // Restart interpreter to ensure threads are not initialized
    py::finalize_interpreter();
    py::initialize_interpreter();
    REQUIRE_FALSE(has_pybind11_internals_static());

    constexpr auto num_threads = 10;
    auto locals = py::dict("count"_a = 0);

    {
        py::gil_scoped_release gil_release{};

        auto threads = std::vector<std::thread>();
        for (auto i = 0; i < num_threads; ++i) {
            threads.emplace_back([&]() {
                py::gil_scoped_acquire gil{};
                locals["count"] = locals["count"].cast<int>() + 1;
            });
        }

        for (auto &thread : threads) {
            thread.join();
        }
    }

    REQUIRE(locals["count"].cast<int>() == num_threads);
}

// Scope exit utility https://stackoverflow.com/a/36644501/7255855
struct scope_exit {
    std::function<void()> f_;
    explicit scope_exit(std::function<void()> f) noexcept : f_(std::move(f)) {}
    ~scope_exit() {
        if (f_) {
            f_();
        }
    }
};

TEST_CASE("Reload module from file") {
    // Disable generation of cached bytecode (.pyc files) for this test, otherwise
    // Python might pick up an old version from the cache instead of the new versions
    // of the .py files generated below
    auto sys = py::module_::import("sys");
    bool dont_write_bytecode = sys.attr("dont_write_bytecode").cast<bool>();
    sys.attr("dont_write_bytecode") = true;
    // Reset the value at scope exit
    scope_exit reset_dont_write_bytecode(
        [&]() { sys.attr("dont_write_bytecode") = dont_write_bytecode; });

    std::string module_name = "test_module_reload";
    std::string module_file = module_name + ".py";

    // Create the module .py file
    std::ofstream test_module(module_file);
    test_module << "def test():\n";
    test_module << "    return 1\n";
    test_module.close();
    // Delete the file at scope exit
    scope_exit delete_module_file([&]() { std::remove(module_file.c_str()); });

    // Import the module from file
    auto module_ = py::module_::import(module_name.c_str());
    int result = module_.attr("test")().cast<int>();
    REQUIRE(result == 1);

    // Update the module .py file with a small change
    test_module.open(module_file);
    test_module << "def test():\n";
    test_module << "    return 2\n";
    test_module.close();

    // Reload the module
    module_.reload();
    result = module_.attr("test")().cast<int>();
    REQUIRE(result == 2);
}

TEST_CASE("sys.argv gets initialized properly") {
    py::finalize_interpreter();
    {
        py::scoped_interpreter default_scope;
        auto module = py::module::import("test_interpreter");
        auto py_widget = module.attr("DerivedWidget")("The question");
        const auto &cpp_widget = py_widget.cast<const Widget &>();
        REQUIRE(cpp_widget.argv0().empty());
    }

    {
        char *argv[] = {strdup("a.out")};
        py::scoped_interpreter argv_scope(true, 1, argv);
        std::free(argv[0]);
        auto module = py::module::import("test_interpreter");
        auto py_widget = module.attr("DerivedWidget")("The question");
        const auto &cpp_widget = py_widget.cast<const Widget &>();
        REQUIRE(cpp_widget.argv0() == "a.out");
    }
    py::initialize_interpreter();
}

TEST_CASE("make_iterator can be called before then after finalizing an interpreter") {
    // Reproduction of issue #2101 (https://github.com/pybind/pybind11/issues/2101)
    py::finalize_interpreter();

    std::vector<int> container;
    {
        pybind11::scoped_interpreter g;
        auto iter = pybind11::make_iterator(container.begin(), container.end());
    }

    REQUIRE_NOTHROW([&]() {
        pybind11::scoped_interpreter g;
        auto iter = pybind11::make_iterator(container.begin(), container.end());
    }());

    py::initialize_interpreter();
}
