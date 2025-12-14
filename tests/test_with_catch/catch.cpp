// The Catch implementation is compiled here. This is a standalone
// translation unit to avoid recompiling it for every test change.

#include <pybind11/embed.h>

// Silence MSVC C++17 deprecation warning from Catch regarding std::uncaught_exceptions (up to
// catch 2.0.1; this should be fixed in the next catch release after 2.0.1).
PYBIND11_WARNING_DISABLE_MSVC(4996)

// Catch uses _ internally, which breaks gettext style defines
#ifdef _
#    undef _
#endif

#define CATCH_CONFIG_RUNNER
#define CATCH_CONFIG_DEFAULT_REPORTER "progress"
#include <catch.hpp>

namespace py = pybind11;

// Simple progress reporter that prints a line per test case.
namespace {

class ProgressReporter : public Catch::CumulativeReporterBase<ProgressReporter> {
public:
    using CumulativeReporterBase::CumulativeReporterBase;

    static std::string getDescription() { return "Simple progress reporter (one line per test)"; }

    void testCaseStarting(Catch::TestCaseInfo const &testInfo) override {
        stream << "[ RUN      ] " << testInfo.name << '\n';
        stream.flush();
        CumulativeReporterBase::testCaseStarting(testInfo);
    }

    void testCaseEnded(Catch::TestCaseStats const &testCaseStats) override {
        auto const &info = testCaseStats.testInfo;
        bool failed = (testCaseStats.totals.assertions.failed > 0);
        stream << (failed ? "[  FAILED  ] " : "[       OK ] ") << info.name << '\n';
        stream.flush();
        CumulativeReporterBase::testCaseEnded(testCaseStats);
    }

    static std::set<Catch::Verbosity> getSupportedVerbosities() {
        return {Catch::Verbosity::Normal};
    }

    void testRunEndedCumulative() override {}

    void noMatchingTestCases(std::string const &spec) override {
        stream << "[  NO TEST ] no matching test cases for spec: " << spec << '\n';
        stream.flush();
    }

    void reportInvalidArguments(std::string const &arg) override {
        stream << "[   ERROR  ] invalid Catch2 arguments: " << arg << '\n';
        stream.flush();
    }
};

} // namespace

CATCH_REGISTER_REPORTER("progress", ProgressReporter)

int main(int argc, char *argv[]) {
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

    py::scoped_interpreter guard{};

    auto result = Catch::Session().run(argc, argv);

    return result < 0xff ? result : 0xff;
}
