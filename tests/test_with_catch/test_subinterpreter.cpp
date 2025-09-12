#include <pybind11/embed.h>
#ifdef PYBIND11_HAS_SUBINTERPRETER_SUPPORT
#    include <pybind11/subinterpreter.h>

// Silence MSVC C++17 deprecation warning from Catch regarding std::uncaught_exceptions (up to
// catch 2.0.1; this should be fixed in the next catch release after 2.0.1).
PYBIND11_WARNING_DISABLE_MSVC(4996)

#    include <catch.hpp>
#    include <cstdlib>
#    include <fstream>
#    include <functional>
#    include <thread>
#    include <utility>

namespace py = pybind11;
using namespace py::literals;

bool has_state_dict_internals_obj();
uintptr_t get_details_as_uintptr();

void unsafe_reset_internals_for_single_interpreter() {
    // NOTE: This code is NOT SAFE unless the caller guarantees no other threads are alive
    // NOTE: This code is tied to the precise implementation of the internals holder

    // first, unref the thread local internals
    py::detail::get_internals_pp_manager().unref();
    py::detail::get_local_internals_pp_manager().unref();

    // we know there are no other interpreters, so we can lower this. SUPER DANGEROUS
    py::detail::get_num_interpreters_seen() = 1;

    // now we unref the static global singleton internals
    py::detail::get_internals_pp_manager().unref();
    py::detail::get_local_internals_pp_manager().unref();

    // finally, we reload the static global singleton
    py::detail::get_internals();
    py::detail::get_local_internals();
}

TEST_CASE("Single Subinterpreter") {
    unsafe_reset_internals_for_single_interpreter();

    py::module_::import("external_module"); // in the main interpreter

    // Add tags to the modules in the main interpreter and test the basics.
    py::module_::import("__main__").attr("main_tag") = "main interpreter";
    {
        auto m = py::module_::import("widget_module");
        m.attr("extension_module_tag") = "added to module in main interpreter";

        REQUIRE(m.attr("add")(1, 2).cast<int>() == 3);
    }
    REQUIRE(has_state_dict_internals_obj());

    auto main_int
        = py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>();

    /// Create and switch to a subinterpreter.
    {
        py::scoped_subinterpreter ssi;

        // The subinterpreter has internals populated
        REQUIRE(has_state_dict_internals_obj());

        py::list(py::module_::import("sys").attr("path")).append(py::str("."));

        auto ext_int
            = py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>();
        py::detail::get_internals();
        REQUIRE(get_details_as_uintptr() == ext_int);
        REQUIRE(ext_int != main_int);

        // Modules tags should be gone.
        REQUIRE_FALSE(py::hasattr(py::module_::import("__main__"), "tag"));
        {
            auto m = py::module_::import("widget_module");
            REQUIRE_FALSE(py::hasattr(m, "extension_module_tag"));

            // Function bindings should still work.
            REQUIRE(m.attr("add")(1, 2).cast<int>() == 3);
        }
    }

    REQUIRE(py::hasattr(py::module_::import("__main__"), "main_tag"));
    REQUIRE(py::hasattr(py::module_::import("widget_module"), "extension_module_tag"));
    REQUIRE(has_state_dict_internals_obj());

    unsafe_reset_internals_for_single_interpreter();
}

#    if PY_VERSION_HEX >= 0x030D0000
TEST_CASE("Move Subinterpreter") {
    std::unique_ptr<py::subinterpreter> sub(new py::subinterpreter(py::subinterpreter::create()));

    // on this thread, use the subinterpreter and import some non-trivial junk
    {
        py::subinterpreter_scoped_activate activate(*sub);

        py::list(py::module_::import("sys").attr("path")).append(py::str("."));
        py::module_::import("datetime");
        py::module_::import("threading");
        py::module_::import("external_module");
    }

    std::thread([&]() {
        // Use it again
        {
            py::subinterpreter_scoped_activate activate(*sub);
            py::module_::import("external_module");
        }
        sub.reset();
    }).join();

    REQUIRE(!sub);

    unsafe_reset_internals_for_single_interpreter();
}
#    endif

TEST_CASE("GIL Subinterpreter") {

    PyInterpreterState *main_interp = PyInterpreterState_Get();

    {
        auto sub = py::subinterpreter::create();

        REQUIRE(main_interp == PyInterpreterState_Get());

        PyInterpreterState *sub_interp = nullptr;

        {
            py::subinterpreter_scoped_activate activate(sub);

            sub_interp = PyInterpreterState_Get();
            REQUIRE(sub_interp != main_interp);

            py::list(py::module_::import("sys").attr("path")).append(py::str("."));
            py::module_::import("datetime");
            py::module_::import("threading");
            py::module_::import("external_module");

            {
                py::subinterpreter_scoped_activate main(py::subinterpreter::main());
                REQUIRE(PyInterpreterState_Get() == main_interp);

                {
                    py::gil_scoped_release nogil{};
                    {
                        py::gil_scoped_acquire yesgil{};
                        REQUIRE(PyInterpreterState_Get() == main_interp);
                    }
                }

                REQUIRE(PyInterpreterState_Get() == main_interp);
            }

            REQUIRE(PyInterpreterState_Get() == sub_interp);

            {
                py::gil_scoped_release nogil{};
                {
                    py::gil_scoped_acquire yesgil{};
                    REQUIRE(PyInterpreterState_Get() == sub_interp);
                }
            }

            REQUIRE(PyInterpreterState_Get() == sub_interp);
        }

        REQUIRE(PyInterpreterState_Get() == main_interp);

        {
            py::gil_scoped_release nogil{};
            {
                py::gil_scoped_acquire yesgil{};
                REQUIRE(PyInterpreterState_Get() == main_interp);
            }
        }

        REQUIRE(PyInterpreterState_Get() == main_interp);

        bool thread_result;

        {
            thread_result = false;
            py::gil_scoped_release nogil{};
            std::thread([&]() {
                {
                    py::subinterpreter_scoped_activate ssa{sub};
                }
                {
                    py::gil_scoped_acquire gil{};
                    thread_result = (PyInterpreterState_Get() == main_interp);
                }
            }).join();
        }
        REQUIRE(thread_result);

        {
            thread_result = false;
            py::gil_scoped_release nogil{};
            std::thread([&]() {
                py::gil_scoped_acquire gil{};
                thread_result = (PyInterpreterState_Get() == main_interp);
            }).join();
        }
        REQUIRE(thread_result);
    }

    REQUIRE(PyInterpreterState_Get() == main_interp);
    unsafe_reset_internals_for_single_interpreter();
}

TEST_CASE("Multiple Subinterpreters") {
    unsafe_reset_internals_for_single_interpreter();

    // Make sure the module is in the main interpreter and save its pointer
    auto *main_ext = py::module_::import("external_module").ptr();
    auto main_int
        = py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>();
    py::module_::import("external_module").attr("multi_interp") = "1";

    {
        py::subinterpreter si1 = py::subinterpreter::create();
        std::unique_ptr<py::subinterpreter> psi2;

        PyObject *sub1_ext = nullptr;
        PyObject *sub2_ext = nullptr;
        uintptr_t sub1_int = 0;
        uintptr_t sub2_int = 0;

        {
            py::subinterpreter_scoped_activate scoped(si1);
            py::list(py::module_::import("sys").attr("path")).append(py::str("."));

            // The subinterpreter has its own copy of this module which is completely separate from
            // main
            sub1_ext = py::module_::import("external_module").ptr();
            REQUIRE(sub1_ext != main_ext);
            REQUIRE_FALSE(py::hasattr(py::module_::import("external_module"), "multi_interp"));
            py::module_::import("external_module").attr("multi_interp") = "2";
            // The subinterpreter also has its own internals
            sub1_int
                = py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>();
            REQUIRE(sub1_int != main_int);

            // while the old one is active, create a new one
            psi2.reset(new py::subinterpreter(py::subinterpreter::create()));
        }

        {
            py::subinterpreter_scoped_activate scoped(*psi2);
            py::list(py::module_::import("sys").attr("path")).append(py::str("."));

            // The second subinterpreter is separate from both main and the other subinterpreter
            sub2_ext = py::module_::import("external_module").ptr();
            REQUIRE(sub2_ext != main_ext);
            REQUIRE(sub2_ext != sub1_ext);
            REQUIRE_FALSE(py::hasattr(py::module_::import("external_module"), "multi_interp"));
            py::module_::import("external_module").attr("multi_interp") = "3";
            // The subinterpreter also has its own internals
            sub2_int
                = py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>();
            REQUIRE(sub2_int != main_int);
            REQUIRE(sub2_int != sub1_int);
        }

        {
            py::subinterpreter_scoped_activate scoped(si1);
            REQUIRE(
                py::cast<std::string>(py::module_::import("external_module").attr("multi_interp"))
                == "2");
        }

        // out here we should be in the main interpreter, with the GIL, with the other 2 still
        // alive

        auto post_int
            = py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>();
        // Make sure internals went back the way it was before
        REQUIRE(main_int == post_int);

        REQUIRE(py::cast<std::string>(py::module_::import("external_module").attr("multi_interp"))
                == "1");
    }

    // now back to just main

    auto post_int
        = py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>();
    // Make sure internals went back the way it was before
    REQUIRE(main_int == post_int);

    REQUIRE(py::cast<std::string>(py::module_::import("external_module").attr("multi_interp"))
            == "1");

    unsafe_reset_internals_for_single_interpreter();
}

#    ifdef Py_MOD_PER_INTERPRETER_GIL_SUPPORTED
TEST_CASE("Per-Subinterpreter GIL") {
    auto main_int
        = py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>();

    std::atomic<int> started, sync, failure;
    started = 0;
    sync = 0;
    failure = 0;

// REQUIRE throws on failure, so we can't use it within the thread
#        define T_REQUIRE(status)                                                                 \
            do {                                                                                  \
                assert(status);                                                                   \
                if (!(status))                                                                    \
                    ++failure;                                                                    \
            } while (0)

    auto &&thread_main = [&](int num) {
        while (started == 0)
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        ++started;

        py::gil_scoped_acquire gil;

        // we have the GIL, we can access the main interpreter
        auto t_int
            = py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>();
        T_REQUIRE(t_int == main_int);
        py::module_::import("external_module").attr("multi_interp") = "1";

        auto sub = py::subinterpreter::create();

        {
            py::subinterpreter_scoped_activate sguard{sub};

            py::list(py::module_::import("sys").attr("path")).append(py::str("."));

            // we have switched to the new interpreter and released the main gil

            // trampoline_module did not provide the per_interpreter_gil tag, so it cannot be
            // imported
            bool caught = false;
            try {
                py::module_::import("trampoline_module");
            } catch (pybind11::error_already_set &pe) {
                T_REQUIRE(pe.matches(PyExc_ImportError));
                std::string msg(pe.what());
                T_REQUIRE(msg.find("does not support loading in subinterpreters")
                          != std::string::npos);
                caught = true;
            }
            T_REQUIRE(caught);

            // widget_module did provide the per_interpreter_gil tag, so it this does not throw
            try {
                py::module_::import("widget_module");
                caught = false;
            } catch (pybind11::error_already_set &) {
                caught = true;
            }
            T_REQUIRE(!caught);

            // widget_module did provide the per_interpreter_gil tag, so it this does not throw
            py::module_::import("widget_module");

            T_REQUIRE(!py::hasattr(py::module_::import("external_module"), "multi_interp"));
            py::module_::import("external_module").attr("multi_interp") = std::to_string(num);

            // wait for something to set sync to our thread number
            // we are holding our subinterpreter's GIL
            while (sync != num)
                std::this_thread::sleep_for(std::chrono::microseconds(1));

            // now change it so the next thread can move on
            ++sync;

            // but keep holding the GIL until after the next thread moves on as well
            while (sync == num + 1)
                std::this_thread::sleep_for(std::chrono::microseconds(1));

            // one last check before quitting the thread, the internals should be different
            auto sub_int
                = py::module_::import("external_module").attr("internals_at")().cast<uintptr_t>();
            T_REQUIRE(sub_int != main_int);
        }
    };
#        undef T_REQUIRE

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

    unsafe_reset_internals_for_single_interpreter();

    // make sure nothing unexpected happened inside the threads, now that they are completed
    REQUIRE(failure == 0);
}
#    endif // Py_MOD_PER_INTERPRETER_GIL_SUPPORTED

#endif // PYBIND11_HAS_SUBINTERPRETER_SUPPORT
