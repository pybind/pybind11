// Fuzz harness: pybind11 class machinery (classes_fuzzer.cpp)
//
// Threat model (see fuzzing-memory-unsafe-expert guidance):
//   * Untrusted input: the fuzz byte stream. In real pybind11 deployments the
//     attacker controls *data crossing the C++<->Python boundary* -- arguments
//     passed to bound functions, attribute names/values, return conversions.
//     This harness models that: the fuzzed bytes drive a bounded sequence of
//     operations (constructor arguments, method arguments, attribute names,
//     wrong-type probes) against a fixed, registered set of C++ fixture types.
//   * Mutated: op sequence + string/int payloads only. The fixture module
//     ("fz") is fixed trusted application code, registered once.
//   * What counts as a real bug: any sanitizer finding (ASan/UBSan) in
//     pybind11's instance lifetime, dispatch, casting or exception
//     translation code reachable from legitimate bound-API use.
//
// Coverage targets (dark in the 2026-11-25 baseline, 10.41% own-code lines):
//   detail/class-inl.h (549 dark, dealloc/new/meta-call paths),
//   detail/type_caster_base{-inl,}.h (832 dark, 0%),
//   pybind11.h cpp_function::dispatcher (322 dark; introspector top unreached
//   complexity), detail/value_and_holder.h (46, 0%),
//   detail/function_record_pyobject.h (83, 0%), detail/init.h,
//   detail/native_enum_data.h (164, 0%), attr.h (144, 0%),
//   detail/exception_translation-inl.h (37, 0%).
//
// Input layout (stable, so crashes reproduce):
//   * Operation opcodes are consumed from the END of the buffer via
//     ConsumeIntegralInRange<uint8_t>, one byte per op, first op = last byte.
//   * String arguments are consumed from the FRONT via
//     ConsumeRandomLengthString(48), in the order the ops need them.
//   * Bounded: at most 40 ops per iteration, at most 8 live objects,
//     strings <= 48 bytes, ints within [-1000, 1000000]. Instances are
//     dropped and gc.collect() runs every iteration so per-iteration leaks
//     and dealloc-order problems surface immediately.

#include "pybind11/pybind11.h"
#include "pybind11/embed.h"
#include "pybind11/native_enum.h"
#include "pybind11/operators.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fuzzer/FuzzedDataProvider.h>

namespace py = pybind11;

namespace pybind11_fuzz {

enum class Color { Red = 1, Green = 2, Blue = 3 };

struct Widget {
    Widget(int v, std::string n) : value(v), label(std::move(n)) {}
    explicit Widget(float f) : value(static_cast<int>(f)), label("float-ctor") {}

    int add(int delta) { return value += delta; }
    std::string name() const { return label; }
    int get_value() const { return value; }
    void set_value(int v) { value = v; }
    std::string get_label() const { return label; }
    void set_label(const std::string &s) { label = s; }
    int fail() const { throw std::runtime_error("fz: deliberate failure"); }
    void set_color(Color c) { color = c; }
    Color get_color() const { return color; }
    std::string color_name() const { return color == Color::Red ? "red" : "not-red"; }
    Widget operator+(const Widget &rhs) const {
        Widget out(value + rhs.value, label + "+" + rhs.label);
        return out;
    }

    int value;
    std::string label;
    Color color = Color::Red;
};

struct Base {
    virtual ~Base() = default;
    virtual std::string who() const { return "Base"; }
};

struct Derived : Base {
    std::string who() const override { return "Derived"; }
    int only_derived() const { return 42; }
};

struct Box {
    explicit Box(std::shared_ptr<Widget> w) : held(std::move(w)) {}
    Widget *get() const { return held.get(); }
    void set(std::shared_ptr<Widget> w) { held = std::move(w); }
    bool has() const { return static_cast<bool>(held); }
    std::shared_ptr<Widget> held;
};

struct Gadget {
    explicit Gadget(int i) : id(i) {}
    int id;
};

static std::unique_ptr<Gadget> make_gadget(int id) { return std::make_unique<Gadget>(id); }
static int gadget_id_copy(Gadget g) { return g.id; }

PYBIND11_EMBEDDED_MODULE(fz, m) {
    py::class_<Widget, std::shared_ptr<Widget>>(m, "Widget", py::dynamic_attr())
        .def(py::init<int, std::string>())
        .def(py::init<float>())
        .def("add", &Widget::add)
        .def("name", &Widget::name)
        .def("fail", &Widget::fail)
        .def("set_color", &Widget::set_color)
        .def("get_color", &Widget::get_color)
        .def("color_name", &Widget::color_name)
        .def_readwrite("value", &Widget::value)
        .def_property("label", &Widget::get_label, &Widget::set_label)
        .def(py::self + py::self)
        .def("__repr__",
             [](const Widget &w) { return "<Widget " + std::to_string(w.value) + ">"; });

    py::class_<Base>(m, "Base").def(py::init<>()).def("who", &Base::who);

    py::class_<Derived, Base>(m, "Derived")
        .def(py::init<>())
        .def("who", &Derived::who)
        .def("only_derived", &Derived::only_derived);

    py::class_<Box>(m, "Box")
        .def(py::init<std::shared_ptr<Widget>>())
        .def("get", &Box::get, py::return_value_policy::reference_internal)
        .def("set", &Box::set)
        .def("has", &Box::has);

    py::class_<Gadget>(m, "Gadget").def(py::init<int>()).def("id", [](const Gadget &g) {
        return g.id;
    });

    m.def("make_gadget", &make_gadget);
    m.def("gadget_id_copy", &gadget_id_copy);

    py::native_enum<Color>(m, "Color", "enum.IntEnum", "fuzz fixture colors")
        .value("Red", Color::Red)
        .value("Green", Color::Green)
        .value("Blue", Color::Blue)
        .finalize();
}

// One-time setup runs from LLVMFuzzerInitialize, before libFuzzer arms its
// per-execution malloc/free tracing for any corpus input. Creating the
// interpreter, importing modules and warming every op here means each traced
// callback execution allocates and frees a balanced set, so libFuzzer's
// in-run leak detection only ever sees real per-iteration leaks (the
// process-lifetime CPython allocations that LSan reports at exit are a known
// property of embedding CPython, also present in the existing pybind_fuzzer).
//
// Destruction order matters: every py::object handle lives inside FuzzEnv,
// declared AFTER the scoped_interpreter member, so member destruction (in
// reverse declaration order) releases all handles BEFORE the interpreter is
// finalized. File-scope static py::object globals would be constructed
// pre-main and therefore destroyed AFTER the guard, causing decrefs on a
// finalized interpreter (SEGV at exit, observed in the rejected run).
struct FuzzEnv {
    py::scoped_interpreter guard;
    py::object fz;
    py::object gc_mod;
    py::object Widget_cls;
    py::object Derived_cls;
    py::object Box_cls;
    py::object Gadget_cls;
    py::object make_gadget_fn;
    py::object gadget_copy_fn;
    py::object Color_cls;

    FuzzEnv()
        : guard{}, fz(py::module_::import("fz")), gc_mod(py::module_::import("gc")),
          Widget_cls(fz.attr("Widget")), Derived_cls(fz.attr("Derived")), Box_cls(fz.attr("Box")),
          Gadget_cls(fz.attr("Gadget")), make_gadget_fn(fz.attr("make_gadget")),
          gadget_copy_fn(fz.attr("gadget_id_copy")), Color_cls(fz.attr("Color")) {
        warm_up();
    }

    static FuzzEnv &get() {
        static FuzzEnv env;
        return env;
    }

    void warm_up() {
        // Exercise every op once with fixed arguments so first-time caches
        // (type method caches, attribute lookups, enum construction) are filled
        // outside traced executions.
        py::object w = Widget_cls(1, std::string("warm"));
        (void) w.attr("add")(1).cast<int>();
        (void) w.attr("name")();
        (void) py::repr(w);
        (void) py::str(w);
        (void) py::hash(w);
        (void) py::bool_(w);
        w.attr("value") = 2;
        (void) w.attr("value").cast<int>();
        w.attr("label") = std::string("warm2");
        (void) w.attr("label").cast<std::string>();
        w.attr("probe") = 3;
        (void) w.attr("probe").cast<int>();
        (void) w.cast<std::shared_ptr<Widget>>();
        py::object box = Box_cls(w);
        (void) box.attr("has")();
        (void) box.attr("get")();
        (void) box.attr("get")().attr("add")(1);
        box.attr("set")(py::none());
        py::object g = make_gadget_fn(7);
        (void) gadget_copy_fn(g);
        try {
            (void) box.attr("add")(1);
        } catch (const py::error_already_set &) {
        }
        try {
            (void) w.attr("fail")();
        } catch (const py::error_already_set &e) {
            (void) e.what();
        }
        py::object val = Color_cls.attr("Red");
        w.attr("set_color")(val);
        (void) w.attr("color_name")();
        (void) w.attr("get_color")().attr("name");
        (void) val.attr("value").cast<int>();
        py::object d = Derived_cls();
        (void) py::isinstance<Derived>(d);
        (void) py::isinstance<Base>(d);
        (void) d.attr("only_derived")().cast<int>();
        // NOTE: no shared_ptr casts here -- Derived is registered with the
        // default (unique_ptr) holder, so loading shared_ptr<Derived/Base> from
        // its instances throws cast_error (rejected-run r4 failure).
        (void) gc_mod.attr("collect")();
    }
};

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
    FuzzEnv::get();
    return 0;
}

// Bounded operation ids. Kept small enough for ConsumeIntegralInRange<uint8_t>.
enum Op : uint8_t {
    kWidgetCtor = 0,
    kWidgetFloatCtor,
    kMethodInt,
    kMethodStr,
    kPropertyRoundtrip,
    kDynamicAttr,
    kReprStrHash,
    kCastBackSharedPtr,
    kBoxOps,
    kGadgetFactory,
    kGadgetCopyArg,
    kWrongTypeCall,
    kRaiseAndCatch,
    kEnumOps,
    kIsinstanceDowncast,
    kDeleteAndGc,
    kMaxOp // sentinel
};

inline uint8_t next_op(FuzzedDataProvider &fdp) {
    return fdp.ConsumeIntegralInRange<uint8_t>(0, static_cast<uint8_t>(kMaxOp - 1));
}

inline int next_int(FuzzedDataProvider &fdp) {
    return fdp.ConsumeIntegralInRange<int>(-1000, 1000000);
}

inline std::string next_str(FuzzedDataProvider &fdp) { return fdp.ConsumeRandomLengthString(48); }

} // namespace pybind11_fuzz

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    using namespace pybind11_fuzz;

    FuzzedDataProvider fdp(data, size);
    std::vector<py::object> live;
    live.reserve(8);

    auto keep = [&live](py::object o) {
        if (live.size() >= 8) {
            live.erase(live.begin());
        }
        live.push_back(std::move(o));
    };

    auto last = [&live]() -> py::object & { return live.back(); };

    try {
        int ops = 0;
        while (fdp.remaining_bytes() > 0 && ops < 40) {
            ++ops;
            const uint8_t op = next_op(fdp);
            try {
                switch (op) {
                    case kWidgetCtor: {
                        int v = next_int(fdp);
                        std::string s = next_str(fdp);
                        keep(FuzzEnv::get().Widget_cls(v, s));
                        break;
                    }
                    case kWidgetFloatCtor: {
                        float f = static_cast<float>(next_int(fdp)) / 7.0f;
                        keep(FuzzEnv::get().Widget_cls(f));
                        break;
                    }
                    case kMethodInt: {
                        if (!live.empty()) {
                            int d = next_int(fdp);
                            (void) last().attr("add")(d).cast<int>();
                        }
                        break;
                    }
                    case kMethodStr: {
                        if (!live.empty()) {
                            std::string s = next_str(fdp);
                            (void) last().attr("name")();
                            (void) last().attr(s.c_str()); // attacker-style attr lookup
                        }
                        break;
                    }

                    case kPropertyRoundtrip: {
                        if (!live.empty()) {
                            int v = next_int(fdp);
                            last().attr("value") = v;
                            (void) last().attr("value").cast<int>();
                            std::string s = next_str(fdp);
                            last().attr("label") = s;
                            (void) last().attr("label").cast<std::string>();
                        }
                        break;
                    }
                    case kDynamicAttr: {
                        if (!live.empty()) {
                            std::string s = next_str(fdp);
                            int v = next_int(fdp);
                            last().attr(s.c_str()) = v;
                            (void) last().attr(s.c_str()).cast<int>();
                        }
                        break;
                    }
                    case kReprStrHash: {
                        if (!live.empty()) {
                            (void) py::repr(last());
                            (void) py::str(last());
                            (void) py::hash(last());
                            (void) py::bool_(last());
                        }
                        break;
                    }
                    case kCastBackSharedPtr: {
                        if (!live.empty()) {
                            (void) py::isinstance<Widget>(last());
                            (void) last().cast<std::shared_ptr<Widget>>();
                        }
                        break;
                    }
                    case kBoxOps: {
                        if (!live.empty()) {
                            py::object box = FuzzEnv::get().Box_cls(last());
                            (void) box.attr("has")();
                            (void) box.attr("get")();
                            (void) box.attr("get")().attr("add")(next_int(fdp));
                            box.attr("set")(py::none());
                            keep(std::move(box));
                        }
                        break;
                    }
                    case kGadgetFactory: {
                        int v = next_int(fdp);
                        keep(FuzzEnv::get().make_gadget_fn(v));
                        break;
                    }
                    case kGadgetCopyArg: {
                        if (!live.empty()) {
                            (void) FuzzEnv::get().gadget_copy_fn(last());
                        }
                        break;
                    }
                    case kWrongTypeCall: {
                        // Dispatch/type-error paths: call methods on a mismatched type.
                        py::object box = FuzzEnv::get().Box_cls(py::none());
                        (void) box.attr("get")(); // None held -> get() returns null ptr
                        try {
                            (void) box.attr("add")(1);
                        } catch (const py::error_already_set &) {
                        }
                        keep(std::move(box));
                        break;
                    }
                    case kRaiseAndCatch: {
                        if (!live.empty()) {
                            try {
                                (void) last().attr("fail")();
                            } catch (const py::error_already_set &e) {
                                (void) e.what(); // exercises error_fetch_and_normalize
                            }
                        }
                        break;
                    }

                    case kEnumOps: {
                        int which = next_op(fdp) % 3;
                        py::object val = FuzzEnv::get().Color_cls.attr(which == 0   ? "Red"
                                                                       : which == 1 ? "Green"
                                                                                    : "Blue");
                        if (!live.empty() && py::isinstance<Widget>(last())) {
                            last().attr("set_color")(val);
                            (void) last().attr("color_name")();
                            (void) last().attr("get_color")().attr("name");
                        }
                        (void) val.attr("value").cast<int>();
                        break;
                    }
                    case kIsinstanceDowncast: {
                        keep(FuzzEnv::get().Derived_cls());
                        if (!live.empty()) {
                            (void) py::isinstance<Derived>(last());
                            (void) py::isinstance<Base>(last());
                            if (py::isinstance<Derived>(last())) {
                                (void) last().attr("only_derived")().cast<int>();
                                (void) last().attr("who")();
                            }
                        }
                        break;
                    }
                    case kDeleteAndGc: {
                        if (!live.empty()) {
                            live.pop_back();
                        }
                        (void) FuzzEnv::get().gc_mod.attr("collect")();
                        break;
                    }
                    default:
                        break;
                }
            } catch (const py::error_already_set &) {
                // Expected control flow: attacker-influenced arguments routinely
                // trigger Python-level exceptions. Sanitizers remain the oracle.
            } catch (const py::builtin_exception &) {
                // Expected control flow: failing casts / unregistered types.
            } catch (const std::exception &) {
                // Keep the gate clean from documented C++ exception paths;
                // memory-safety violations crash before this handler.
            }
        }
    } catch (const py::error_already_set &) {
    } catch (const py::builtin_exception &) {
    } catch (const std::exception &) {
    }

    live.clear();
    try {
        (void) FuzzEnv::get().gc_mod.attr("collect")();
    } catch (const py::error_already_set &) {
    } catch (const py::builtin_exception &) {
    }
    return 0;
}
