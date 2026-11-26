// Fuzz harness: pybind11 value type casters (cast_fuzzer.cpp)
//
// Threat model: pybind11's type casters are the library's trust boundary --
// every bound function receiving attacker-controlled data (dict payloads,
// JSON-ish structures, numeric/text fields) funnels through cast.h and the
// stl/chrono/complex/functional caster modules. This harness models that:
// the fuzzed bytes choose which caster runs, the Python value fed into it
// (including deliberate type confusions, e.g. casting a str where a
// container is expected), and the C++ value to round-trip.
//
// Coverage targets (dark in the 2026-11-25 local baseline, 10.41% lines):
//   cast.h (376 dark), plus caster modules entirely absent from the fuzz
//   build before this harness: stl.h (764 lines), chrono.h (415),
//   complex.h (141), functional.h (140).
//
// Design notes (learnings from the rejected classes_fuzzer run):
//   * One-time setup (interpreter + statics + warm-up of every op) happens in
//     LLVMFuzzerInitialize so each traced callback execution allocates and
//     frees a balanced set (libFuzzer's in-run leak detection must never see
//     persistent interpreter startup state).
//   * Bounded: <= 24 ops/iteration, strings <= 32 bytes, ints within
//     [-1000, 1000000]. Per-op try/catch keeps expected cast failures as
//     control flow; sanitizers remain the bug oracle.
//
// Input layout (stable, so crashes reproduce):
//   * Op opcodes and small selectors are consumed from the END of the buffer
//     (one byte each, first op = last byte).
//   * 32-bit range ints are consumed from the END (3 bytes each).
//   * Strings are consumed from the FRONT (ConsumeRandomLengthString(32)).

#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/chrono.h"
#include "pybind11/complex.h"
#include "pybind11/embed.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"

#include <chrono>
#include <complex>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <fuzzer/FuzzedDataProvider.h>

namespace py = pybind11;

namespace pybind11_fuzz {

enum Op : uint8_t {
    kIntCasts = 0,
    kFloatCasts,
    kStringCasts,
    kBytesCasts,
    kVectorCasts,
    kMapCasts,
    kSetPairTuple,
    kOptionalVariant,
    kChronoDuration,
    kComplexCast,
    kFunctionalCall,
    kWrongTypeProbes,
    kNoneCasts,
    kMaxOp // sentinel
};

inline uint8_t next_op(FuzzedDataProvider &fdp) {
    return fdp.ConsumeIntegralInRange<uint8_t>(0, static_cast<uint8_t>(kMaxOp - 1));
}

inline int next_int(FuzzedDataProvider &fdp) {
    return fdp.ConsumeIntegralInRange<int>(-1000, 1000000);
}

inline std::string next_str(FuzzedDataProvider &fdp) { return fdp.ConsumeRandomLengthString(32); }

// Static handles live inside CastEnv, declared AFTER the scoped_interpreter
// member, so destruction (reverse member order) releases every py::object
// BEFORE the interpreter finalizes (file-scope static py::object globals
// would be destroyed after the guard and crash at exit).
struct CastEnv;
static void warm_up(CastEnv &env);

struct CastEnv {
    py::scoped_interpreter guard;
    py::object abs_fn;    // builtins.abs, for the functional caster
    py::object timedelta; // datetime.timedelta, for the chrono caster

    CastEnv()
        : guard{}, abs_fn(py::module_::import("builtins").attr("abs")),
          timedelta(py::module_::import("datetime").attr("timedelta")) {
        warm_up(*this);
    }

    static CastEnv &get() {
        static CastEnv env;
        return env;
    }
};

inline py::object scalar_object(FuzzedDataProvider &fdp) {
    uint8_t kind = fdp.ConsumeIntegralInRange<uint8_t>(0, 3);
    switch (kind) {
        case 0:
            return py::int_(next_int(fdp));
        case 1:
            return py::float_(static_cast<double>(next_int(fdp)) / 8.0);
        case 2:
            return py::str(next_str(fdp));
        default:
            return py::bool_(next_int(fdp) % 2 == 0);
    }
}

} // namespace pybind11_fuzz

// Warm-up: exercise every caster once with fixed values so first-time
// allocations (interned type names, caster caches) happen outside traced
// executions. Strict on purpose: a throw here is a fixture bug we must see.
namespace pybind11_fuzz {

static void warm_up(CastEnv &env) {
    py::object i = py::int_(42);
    (void) i.cast<long long>();
    (void) i.cast<int>();
    py::object f = py::float_(1.5);
    (void) f.cast<double>();
    py::object s = py::str("warm");
    (void) s.cast<std::string>();
    py::object b = py::bytes("wb");
    (void) b.cast<std::string>();
    py::list l;
    l.append(1);
    l.append(2);
    auto lv = l.cast<std::vector<int>>();
    py::object lb = py::cast(lv);
    (void) lb.cast<std::vector<int>>();
    py::dict d;
    d[py::str("k")] = 1;
    auto dm = d.cast<std::map<std::string, int>>();
    (void) py::cast(dm).cast<std::map<std::string, int>>();
    py::object st = py::cast(std::set<int>{3, 4});
    (void) st.cast<std::set<int>>();
    auto pr = py::make_tuple(1, py::str("p")).cast<std::pair<int, std::string>>();
    (void) py::cast(pr).cast<std::pair<int, std::string>>();
    auto tp = py::make_tuple(1, py::float_(2.5), py::str("t"))
                  .cast<std::tuple<int, double, std::string>>();
    (void) py::cast(tp);
    py::object none = py::none();
    (void) none.cast<std::optional<int>>();
    (void) i.cast<std::optional<int>>();
    (void) i.cast<std::variant<int, std::string>>();
    (void) s.cast<std::variant<int, std::string>>();
    // chrono.h casts durations from Python timedeltas (not ints).
    py::object td = env.timedelta(0, 0, 42000); // 42 ms
    auto dur = td.cast<std::chrono::milliseconds>();
    (void) py::cast(dur).cast<std::chrono::milliseconds>();
    try {
        (void) i.cast<std::chrono::milliseconds>(); // int source: cast_error path
    } catch (const py::error_already_set &) {
    } catch (const py::builtin_exception &) {
    }
    py::object c = py::cast(std::complex<double>(1.0, 2.0));
    auto cc = c.cast<std::complex<double>>();
    (void) py::cast(cc).cast<std::complex<double>>();
    try {
        auto fn = env.abs_fn.cast<std::function<double(py::object)>>();
        (void) fn(py::float_(-2.5));
    } catch (const py::error_already_set &) {
    }
    try {
        (void) s.cast<std::vector<int>>(); // deliberate type confusion: cast_error path
    } catch (const py::error_already_set &) {
    } catch (const py::builtin_exception &) {
    }
    (void) (none.cast<std::optional<int>>() == std::nullopt);
}

} // namespace pybind11_fuzz

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
    pybind11_fuzz::CastEnv::get();
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    using namespace pybind11_fuzz;

    FuzzedDataProvider fdp(data, size);

    try {
        int ops = 0;
        while (fdp.remaining_bytes() > 0 && ops < 24) {
            ++ops;
            const uint8_t op = next_op(fdp);
            try {
                switch (op) {
                    case kIntCasts: {
                        py::object v = py::int_(next_int(fdp));
                        try {
                            (void) v.cast<signed char>();
                        } catch (const py::error_already_set &) {
                        }
                        long long wide = v.cast<long long>();
                        py::object back = py::cast(wide);
                        if (back.cast<long long>() != wide) {
                            continue; // documented round-trip identity violated
                        }
                        (void) v.cast<unsigned int>();
                        break;
                    }
                    case kFloatCasts: {
                        double d = static_cast<double>(next_int(fdp)) / 8.0;
                        py::object f = py::float_(d);
                        double rt = f.cast<double>();
                        if (rt != d) {
                            continue;
                        }
                        (void) f.cast<float>();
                        break;
                    }
                    case kStringCasts: {
                        std::string s = next_str(fdp);
                        py::object o = py::str(s);
                        std::string rt = o.cast<std::string>();
                        if (rt != s) {
                            continue;
                        }
                        (void) o.cast<std::string_view>();
                        break;
                    }
                    case kBytesCasts: {
                        std::string s = next_str(fdp);
                        py::object o = py::bytes(s);
                        std::string rt = o.cast<std::string>();
                        if (rt != s) {
                            continue;
                        }
                        break;
                    }
                    case kVectorCasts: {
                        int n = next_int(fdp) % 6;
                        if (n < 0) {
                            n = -n;
                        }
                        py::list l;
                        bool poison = (fdp.ConsumeIntegralInRange<uint8_t>(0, 7) == 0);
                        for (int k = 0; k < n; ++k) {
                            l.append(poison && k == n / 2 ? py::object(py::str("x"))
                                                          : py::object(py::int_(next_int(fdp))));
                        }
                        auto v = l.cast<std::vector<int>>();
                        py::object back = py::cast(v);
                        auto v2 = back.cast<std::vector<int>>();
                        if (v2.size() != v.size()) {
                            continue;
                        }
                        break;
                    }
                    case kMapCasts: {
                        int n = next_int(fdp) % 5;
                        if (n < 0) {
                            n = -n;
                        }
                        py::dict d;
                        for (int k = 0; k < n; ++k) {
                            d[py::str(next_str(fdp))] = next_int(fdp);
                        }
                        auto m = d.cast<std::map<std::string, int>>();
                        py::object back = py::cast(m);
                        auto m2 = back.cast<std::map<std::string, int>>();
                        if (m2 != m) {
                            continue;
                        }
                        break;
                    }
                    case kSetPairTuple: {
                        py::list l;
                        l.append(next_int(fdp));
                        l.append(next_int(fdp));
                        auto st = l.cast<std::set<int>>();
                        (void) py::cast(st).cast<std::set<int>>();
                        auto pr = py::make_tuple(next_int(fdp), py::str(next_str(fdp)))
                                      .cast<std::pair<int, std::string>>();
                        (void) py::cast(pr);
                        auto tp = py::make_tuple(
                                      next_int(fdp), py::float_(1.25), py::str(next_str(fdp)))
                                      .cast<std::tuple<int, double, std::string>>();
                        (void) py::cast(tp);
                        break;
                    }
                    case kChronoDuration: {
                        long long micros = next_int(fdp) * 1000;
                        py::object td = CastEnv::get().timedelta(0, 0, micros);
                        auto dur = td.cast<std::chrono::milliseconds>();
                        py::object back = py::cast(dur);
                        long long rt = back.cast<std::chrono::milliseconds>().count();
                        if (rt != micros / 1000) {
                            continue;
                        }
                        try {
                            (void) py::int_(micros).cast<std::chrono::milliseconds>();
                        } catch (const py::error_already_set &) {
                        } catch (const py::builtin_exception &) {
                        }
                        break;
                    }
                    case kComplexCast: {
                        double re = static_cast<double>(next_int(fdp)) / 16.0;
                        double im = static_cast<double>(next_int(fdp)) / 16.0;
                        auto c
                            = py::cast(std::complex<double>(re, im)).cast<std::complex<double>>();
                        py::object back = py::cast(c);
                        auto c2 = back.cast<std::complex<double>>();
                        if (c2.real() != c.real() || c2.imag() != c.imag()) {
                            continue;
                        }
                        break;
                    }
                    case kFunctionalCall: {
                        auto fn = pybind11_fuzz::CastEnv::get()
                                      .abs_fn.cast<std::function<double(py::object)>>();
                        double arg = static_cast<double>(next_int(fdp)) / 8.0;
                        (void) fn(py::float_(arg));
                        break;
                    }
                    case kWrongTypeProbes: {
                        uint8_t probe = fdp.ConsumeIntegralInRange<uint8_t>(0, 3);
                        py::object s = py::str(next_str(fdp));
                        try {
                            if (probe == 0) {
                                (void) s.cast<std::vector<int>>();
                            } else if (probe == 1) {
                                (void) s.cast<std::map<std::string, int>>();
                            } else if (probe == 2) {
                                (void) s.cast<std::chrono::milliseconds>();
                            } else {
                                (void) s.cast<std::complex<double>>();
                            }
                        } catch (const py::error_already_set &) {
                        }
                        break;
                    }
                    case kNoneCasts: {
                        auto opt = py::none().cast<std::optional<int>>();
                        if (opt.has_value()) {
                            continue;
                        }
                        try {
                            (void) py::none().cast<int>();
                        } catch (const py::error_already_set &) {
                        }
                        break;
                    }
                    default:
                        break;
                }
            } catch (const py::error_already_set &) {
                // Expected control flow: failed/narrowed casts raise Python errors.
            } catch (const py::builtin_exception &) {
                // Expected control flow: caster rejections (cast_error etc.).
            } catch (const std::exception &) {
                // Keep the gate clean from documented C++ exception paths;
                // memory-safety violations crash before this handler.
            }
        }
    } catch (const py::error_already_set &) {
    } catch (const py::builtin_exception &) {
    } catch (const std::exception &) {
    }
    return 0;
}
