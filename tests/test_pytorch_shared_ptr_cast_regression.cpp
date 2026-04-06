#include "pybind11_tests.h"

#include <memory>
#include <string>

#if defined(__clang__)
#    pragma clang diagnostic error "-Wdeprecated-copy-with-user-provided-dtor"
#    pragma clang diagnostic error "-Wdeprecated-copy-with-dtor"
#endif

namespace test_pytorch_regressions {

// Directly extracted from PyTorch patterns that regressed in CI.
struct TracingState : std::enable_shared_from_this<TracingState> {
    TracingState() = default;
    ~TracingState() = default;
    int value = 0;
};

const std::shared_ptr<TracingState> &get_tracing_state() {
    static std::shared_ptr<TracingState> state = std::make_shared<TracingState>();
    return state;
}

struct InterfaceType {
    ~InterfaceType() = default;
    int value = 0;
};
using InterfaceTypePtr = std::shared_ptr<InterfaceType>;

struct CompilationUnit {
    InterfaceTypePtr iface = std::make_shared<InterfaceType>();

    InterfaceTypePtr get_interface(const std::string &) const { return iface; }
};

} // namespace test_pytorch_regressions

TEST_SUBMODULE(pybind11_pytorch_regressions, m) {
    using namespace test_pytorch_regressions;

    py::class_<TracingState, std::shared_ptr<TracingState>>(m, "TracingState")
        .def(py::init<>())
        .def_readwrite("value", &TracingState::value);

    m.def("_get_tracing_state", []() { return get_tracing_state(); });

    py::class_<InterfaceType, InterfaceTypePtr>(m, "InterfaceType")
        .def(py::init<>())
        .def_readwrite("value", &InterfaceType::value);

    py::class_<CompilationUnit, std::shared_ptr<CompilationUnit>>(m, "CompilationUnit")
        .def(py::init<>())
        .def("get_interface",
             [](const std::shared_ptr<CompilationUnit> &self, const std::string &name) {
                 return self->get_interface(name);
             });
}
