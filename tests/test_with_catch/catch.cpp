// The Catch implementation is compiled here. This is a standalone
// translation unit to avoid recompiling it for every test change.

#include <pybind11/embed.h>

#include <chrono>
#include <csignal>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>

#ifndef _WIN32
#    include <unistd.h>
#endif

// Silence MSVC C++17 deprecation warning from Catch regarding std::uncaught_exceptions (up to
// catch 2.0.1; this should be fixed in the next catch release after 2.0.1).
PYBIND11_WARNING_DISABLE_MSVC(4996)

// Catch uses _ internally, which breaks gettext style defines
#ifdef _
#    undef _
#endif

#define CATCH_CONFIG_RUNNER
#define CATCH_CONFIG_DEFAULT_REPORTER "progress"
#include "catch_skip.h"

#include <catch.hpp>

namespace py = pybind11;

// Simple progress reporter that prints a line per test case.
namespace {

class ProgressReporter : public Catch::StreamingReporterBase<ProgressReporter> {
public:
    using StreamingReporterBase<ProgressReporter>::StreamingReporterBase;

    static std::string getDescription() { return "Simple progress reporter (one line per test)"; }

    void testCaseStarting(Catch::TestCaseInfo const &testInfo) override {
        print_python_version_once();
        auto &os = Catch::cout();
        os << "[ RUN      ] " << testInfo.name << '\n';
        os.flush();
    }

    void testCaseEnded(Catch::TestCaseStats const &stats) override {
        bool failed = stats.totals.assertions.failed > 0;
        auto &os = Catch::cout();
        os << (failed ? "[  FAILED  ] " : "[       OK ] ") << stats.testInfo.name << '\n';
        os.flush();
    }

    void noMatchingTestCases(std::string const &spec) override {
        auto &os = Catch::cout();
        os << "[  NO TEST ] no matching test cases for spec: " << spec << '\n';
        os.flush();
    }

    void reportInvalidArguments(std::string const &arg) override {
        auto &os = Catch::cout();
        os << "[   ERROR  ] invalid Catch2 arguments: " << arg << '\n';
        os.flush();
    }

    void assertionStarting(Catch::AssertionInfo const &) override {}

    bool assertionEnded(Catch::AssertionStats const &) override { return false; }

    void testRunEnded(Catch::TestRunStats const &stats) override {
        auto &os = Catch::cout();
        auto passed = stats.totals.testCases.passed;
        auto failed = stats.totals.testCases.failed;
        auto total = passed + failed;
        auto assertions = stats.totals.assertions.passed + stats.totals.assertions.failed;
        if (failed == 0) {
            os << "[  PASSED  ] " << total << " test cases, " << assertions << " assertions.\n";
        } else {
            os << "[  FAILED  ] " << failed << " of " << total << " test cases, " << assertions
               << " assertions.\n";
        }
        os.flush();
    }

private:
    void print_python_version_once() {
        if (printed_) {
            return;
        }
        printed_ = true;
        auto &os = Catch::cout();
        os << "[ PYTHON   ] " << Py_GetVersion() << '\n';
        os.flush();
    }

    bool printed_ = false;
};

} // namespace

CATCH_REGISTER_REPORTER("progress", ProgressReporter)

namespace {

std::string get_utc_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::tm utc_tm{};
#if defined(_WIN32)
    gmtime_s(&utc_tm, &time_t_now);
#else
    gmtime_r(&time_t_now, &utc_tm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&utc_tm, "%Y-%m-%d %H:%M:%S") << '.' << std::setfill('0') << std::setw(3)
        << ms.count() << 'Z';
    return oss.str();
}

#ifndef _WIN32
// Signal handler to print a message when the process is terminated.
// Uses only async-signal-safe functions.
void termination_signal_handler(int sig) {
    const char *msg = "[  SIGNAL  ] Process received SIGTERM\n";
    // write() is async-signal-safe, unlike std::cout
    ssize_t written = write(STDOUT_FILENO, msg, strlen(msg));
    (void) written; // suppress "unused variable" warnings
    // Re-raise with default handler to get proper exit status
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}
#endif

} // namespace

int main(int argc, char *argv[]) {
#ifndef _WIN32
    std::signal(SIGTERM, termination_signal_handler);
#endif

    // Setup for TEST_CASE in test_interpreter.cpp, tagging on a large random number:
    std::string updated_pythonpath("pybind11_test_with_catch_PYTHONPATH_2099743835476552");
    const char *preexisting_pythonpath = getenv("PYTHONPATH");
    if (preexisting_pythonpath != nullptr) {
#if defined(_WIN32)
        updated_pythonpath += ';';
#else
        updated_pythonpath += ':';
#endif
        updated_pythonpath += preexisting_pythonpath;
    }
#if defined(_WIN32)
    _putenv_s("PYTHONPATH", updated_pythonpath.c_str());
#else
    setenv("PYTHONPATH", updated_pythonpath.c_str(), /*replace=*/1);
#endif

    std::cout << "[ STARTING ] " << get_utc_timestamp() << '\n';
    std::cout.flush();

    py::scoped_interpreter guard{};

    auto result = Catch::Session().run(argc, argv);

    std::cout << "[ DONE     ] " << get_utc_timestamp() << " (result " << result << ")\n";
    std::cout.flush();

    return result < 0xff ? result : 0xff;
}
