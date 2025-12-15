// Minimal CPython 3.14 free-threading Move Subinterpreter redux.
//
// This version is intentionally modeled more closely after pybind11's
// `subinterpreter` + "Move Subinterpreter" test:
//
//  - Create a subinterpreter via Py_NewInterpreterFromConfig with
//    PyInterpreterConfig_OWN_GIL and allow_threads=1.
//  - On the main thread, temporarily activate the subinterpreter and
//    import some non-trivial modules.
//  - On a worker thread, activate the same subinterpreter again, run
//    some code, and then *destroy* the subinterpreter from that thread
//    using Py_EndInterpreter with a fresh PyThreadState created on
//    that thread (mirroring pybind11's destructor on 3.13+).
//
// Critical differences from the original pybind11 test:
//  - We do not keep a permanent PyThreadState* for the subinterpreter.
//    Each thread creates a temporary thread state while it is using
//    the subinterpreter, then clears and deletes it again (similar
//    to pybind11::subinterpreter_scoped_activate).
//  - When destroying the subinterpreter we create a new thread state
//    on the worker thread and pass that to Py_EndInterpreter, with no
//    other live thread states for that interpreter, matching the
//    intended CPython contract for Py_EndInterpreter.
//
// Build against a free-threaded CPython 3.14 installation using the
// accompanying shell script.

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Global handle to the subinterpreter's state (mirrors pybind11::subinterpreter::istate_).
static PyInterpreterState *sub_interp = NULL;

static void fatal(const char *msg) {
    fprintf(stderr, "FATAL: %s\n", msg);
    fflush(stderr);
    exit(1);
}

// Helper: run some Python code inside the subinterpreter on the current thread,
// creating a temporary PyThreadState and then cleaning it up again.
static void run_in_subinterpreter(const char *label, const char *code) {
    if (sub_interp == NULL) {
        fatal("run_in_subinterpreter called with sub_interp == NULL");
    }

    PyThreadState *tstate = PyThreadState_New(sub_interp);
    if (tstate == NULL) {
        fatal("PyThreadState_New failed in run_in_subinterpreter");
    }

    fprintf(stderr, "%s: activating subinterpreter on this thread\n", label);
    PyThreadState_Swap(tstate);

    if (PyRun_SimpleString(code) != 0) {
        PyErr_Print();
        fatal("PyRun_SimpleString failed in subinterpreter");
    }

    fprintf(stderr, "%s: finished running code in subinterpreter\n", label);

    // Clean up the temporary thread state. After this, the current thread
    // no longer has an active thread state for any interpreter.
    PyThreadState_Clear(tstate);
    PyThreadState_DeleteCurrent();
}

// Helper: create the subinterpreter with a configuration similar to
// pybind11::subinterpreter::create().
static void create_subinterpreter(void) {
    if (sub_interp != NULL) {
        fatal("create_subinterpreter called twice");
    }

    PyInterpreterConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.allow_threads = 1;
    cfg.check_multi_interp_extensions = 1;
    cfg.gil = PyInterpreterConfig_OWN_GIL;

    PyThreadState *creation_tstate = NULL;
    PyStatus status = Py_NewInterpreterFromConfig(&creation_tstate, &cfg);
    if (PyStatus_Exception(status)) {
        Py_ExitStatusException(status);
    }
    if (creation_tstate == NULL || creation_tstate->interp == NULL) {
        fatal("Py_NewInterpreterFromConfig returned NULL interpreter");
    }

    sub_interp = creation_tstate->interp;

    // On 3.13+ pybind11 clears and deletes the creation thread state right away.
#if PY_VERSION_HEX >= 0x030D0000
    PyThreadState_Clear(creation_tstate);
    PyThreadState_DeleteCurrent();
#endif

    fprintf(stderr, "Subinterpreter created.\n");
}

// Helper: destroy the subinterpreter from the current thread, mirroring
// pybind11::subinterpreter::~subinterpreter() on 3.13+.
static void destroy_subinterpreter_from_current_thread(const char *label) {
    if (sub_interp == NULL) {
        fatal("destroy_subinterpreter_from_current_thread called with sub_interp == NULL");
    }

    PyThreadState *destroy_tstate = PyThreadState_New(sub_interp);
    if (destroy_tstate == NULL) {
        fatal("PyThreadState_New failed in destroy_subinterpreter_from_current_thread");
    }

    PyThreadState *old_tstate = PyThreadState_Swap(destroy_tstate);

    fprintf(stderr, "%s: calling Py_EndInterpreter on subinterpreter\n", label);
    Py_EndInterpreter(destroy_tstate);
    fprintf(stderr, "%s: returned from Py_EndInterpreter\n", label);

    // If there was a previous thread state belonging to a different interpreter,
    // restore it (this should normally be the main interpreter).
    if (old_tstate != NULL && old_tstate->interp != sub_interp) {
        PyThreadState_Swap(old_tstate);
    }

    sub_interp = NULL;
}

static void *worker_thread(void *arg) {
    (void) arg;

    // Use the subinterpreter again from this worker thread.
    run_in_subinterpreter("worker",
                          "import datetime\n"
                          "import threading\n"
                          "print('worker: ran code in subinterpreter')\n");

    // Now destroy the subinterpreter from this worker thread.
    destroy_subinterpreter_from_current_thread("worker");

    return NULL;
}

int main(int argc, char **argv) {
    (void) argc;
    (void) argv;

    // Initialize the main interpreter.
    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.isolated = 0;
    config.install_signal_handlers = 0;

    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status)) {
        Py_ExitStatusException(status);
    }

    // First line of output: the Python version.
    fprintf(stderr, "Python version: %s\n", Py_GetVersion());

    PyThreadState *main_tstate = PyThreadState_Get();
    if (main_tstate == NULL) {
        fatal("PyThreadState_Get returned NULL");
    }

    fprintf(stderr, "Main interpreter initialized.\n");

    // Create a subinterpreter with its own GIL, similar to pybind11::subinterpreter::create().
    create_subinterpreter();

    // On the main thread, activate the subinterpreter and import some modules.
    run_in_subinterpreter("main",
                          "import sys\n"
                          "import datetime\n"
                          "import threading\n"
                          "print('main: ran code in subinterpreter')\n");

    fprintf(stderr, "Subinterpreter imports on main thread done.\n");

    // Start a worker thread that uses the same subinterpreter and then destroys it.
    pthread_t th;
    if (pthread_create(&th, NULL, worker_thread, NULL) != 0) {
        fatal("pthread_create failed");
    }

    if (pthread_join(th, NULL) != 0) {
        fatal("pthread_join failed");
    }

    fprintf(stderr, "Worker thread joined.\n");

    // At this point the subinterpreter should be gone. Finalize the main interpreter.
    PyThreadState_Swap(main_tstate);
    int rc = Py_FinalizeEx();
    fprintf(stderr, "Py_FinalizeEx() returned %d.\n", rc);

    return (rc == 0) ? 0 : 1;
}
