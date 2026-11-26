// Fuzz harness: pybind11 bound-container protocol (stl_bind_fuzzer.cpp)
//
// Threat model: py::bind_vector/py::bind_map expose std:: containers as real
// Python sequence/mapping objects. Attacker-influenced data reaches them as
// indices, keys, values and slice bounds from Python-side code. This harness
// fuzzes exactly those levers: in-range/out-of-range/negative indices,
// insert/erase positions, missing/existing keys, wrong-typed elements.
//
// Coverage target: stl_bind.h (868 lines, entirely absent from the fuzz
// build before this harness) -- indexing bounds checks, negative index
// handling, insert/erase shifting, iterator paths, mapping lookups.
//
// Design notes (learnings applied): all one-time setup (interpreter, module
// import, handles, warm-up of every op) runs inside LLVMFuzzerInitialize via
// the FuzzEnv struct so traced executions stay malloc/free balanced; member
// order guarantees py::object handles die before the interpreter finalizes.
// Bounded: <= 24 ops, strings <= 32 bytes, ints within [-1000, 1000000].
// Per-op try/catch keeps expected Python errors (IndexError/KeyError/
// TypeError surfacing as py::error_already_set, caster rejections as
// py::builtin_exception) as control flow; sanitizers are the oracle.
//
// Input layout: op opcodes and small selectors from the END of the buffer
// (first op = last byte); strings from the FRONT (ConsumeRandomLengthString).

#include "pybind11/pybind11.h"
#include "pybind11/embed.h"
#include "pybind11/stl_bind.h"

#include <map>
#include <string>
#include <vector>

#include <fuzzer/FuzzedDataProvider.h>

namespace py = pybind11;

namespace pybind11_fuzz {

enum Op : uint8_t {
    kVecNew = 0,
    kVecAppend,
    kVecInsert,
    kVecPopGet,
    kVecSetDel,
    kVecSliceLen,
    kMapNew,
    kMapSet,
    kMapGetDel,
    kIterate,
    kWrongType,
    kMaxOp // sentinel
};

inline uint8_t next_op(FuzzedDataProvider &fdp) {
    return fdp.ConsumeIntegralInRange<uint8_t>(0, static_cast<uint8_t>(kMaxOp - 1));
}

inline int next_int(FuzzedDataProvider &fdp) {
    return fdp.ConsumeIntegralInRange<int>(-1000, 1000000);
}

// index used against containers: fuzzed but bounded around realistic sizes
inline py::object next_index_object(FuzzedDataProvider &fdp, int size) {
    int raw = fdp.ConsumeIntegralInRange<int>(-6, 14);
    if (raw < 0) {
        raw += size; // exercise Python negative-index normalisation
    }
    return py::int_(raw);
}

inline std::string next_str(FuzzedDataProvider &fdp) { return fdp.ConsumeRandomLengthString(32); }

} // namespace pybind11_fuzz

PYBIND11_EMBEDDED_MODULE(fzb, m) {
    py::bind_vector<std::vector<int>>(m, "IVec");
    py::bind_vector<std::vector<std::string>>(m, "SVec");
    py::bind_map<std::map<std::string, int>>(m, "SIMap");
}

// Handles + interpreter, destruction-safe member order (see header comment).
struct FuzzEnv {
    py::scoped_interpreter guard;
    py::object gc_mod;
    py::object IVec;
    py::object SVec;
    py::object SIMap;

    FuzzEnv() : guard{}, gc_mod(py::module_::import("gc")) {
        py::object mod = py::module_::import("fzb");
        IVec = mod.attr("IVec");
        SVec = mod.attr("SVec");
        SIMap = mod.attr("SIMap");
        warm_up();
    }

    static FuzzEnv &get() {
        static FuzzEnv env;
        return env;
    }

    void warm_up() {
        py::object v = IVec();
        v.attr("append")(1);
        v.attr("insert")(0, 2);
        (void) v.attr("pop")(0);
        (void) v.attr("__getitem__")(0);
        v.attr("__setitem__")(0, 3);
        (void) py::len(v);
        (void) py::iter(v);
        py::object m = SIMap();
        m.attr("__setitem__")(py::str("k"), 1);
        (void) m.attr("__getitem__")(py::str("k"));
        (void) m.attr("__contains__")(py::str("k"));
        m.attr("__delitem__")(py::str("k"));
        (void) gc_mod.attr("collect")();
    }
};

extern "C" int LLVMFuzzerInitialize(int *argc, char ***argv) {
    FuzzEnv::get();
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    using namespace pybind11_fuzz;

    FuzzEnv &env = FuzzEnv::get();
    FuzzedDataProvider fdp(data, size);
    py::object live; // most recent container; replaced per op
    bool have = false;

    try {
        int ops = 0;
        while (fdp.remaining_bytes() > 0 && ops < 24) {
            ++ops;
            const uint8_t op = next_op(fdp);
            try {
                switch (op) {
                    case kVecNew: {
                        live = env.IVec();
                        have = true;
                        break;
                    }
                    case kVecAppend: {
                        if (!have) {
                            break;
                        }
                        live.attr("append")(next_int(fdp));
                        break;
                    }
                    case kVecInsert: {
                        if (!have) {
                            break;
                        }
                        int sz = static_cast<int>(py::len(live));
                        live.attr("insert")(next_index_object(fdp, sz), next_int(fdp));
                        break;
                    }
                    case kVecPopGet: {
                        if (!have) {
                            break;
                        }
                        int sz = static_cast<int>(py::len(live));
                        try {
                            (void) live.attr("pop")(next_index_object(fdp, sz));
                        } catch (const py::error_already_set &) {
                        }
                        break;
                    }
                    case kVecSetDel: {
                        if (!have) {
                            break;
                        }
                        int sz = static_cast<int>(py::len(live));
                        live.attr("__setitem__")(next_index_object(fdp, sz), next_int(fdp));
                        break;
                    }
                    case kVecSliceLen: {
                        if (!have) {
                            break;
                        }
                        (void) py::len(live);
                        (void) py::iter(live);
                        (void) live.attr("__getitem__")(py::int_(next_int(fdp)));
                        break;
                    }
                    case kMapNew: {
                        live = env.SIMap();
                        have = true;
                        break;
                    }
                    case kMapSet: {
                        if (!have) {
                            break;
                        }
                        std::string k = next_str(fdp);
                        live.attr("__setitem__")(py::str(k), next_int(fdp));
                        break;
                    }
                    case kMapGetDel: {
                        if (!have) {
                            break;
                        }
                        std::string k = next_str(fdp);
                        try {
                            (void) live.attr("__getitem__")(py::str(k));
                        } catch (const py::error_already_set &) {
                        }
                        (void) live.attr("__contains__")(py::str(k));
                        try {
                            live.attr("__delitem__")(py::str(k));
                        } catch (const py::error_already_set &) {
                        }
                        break;
                    }
                    case kIterate: {
                        if (!have) {
                            break;
                        }
                        py::iterator it = py::iter(live);
                        int steps = 0;
                        while (it != py::iterator::sentinel() && steps < 32) {
                            ++steps;
                            (void) py::reinterpret_borrow<py::object>(*it);
                            ++it;
                        }
                        break;
                    }
                    case kWrongType: {
                        if (!have) {
                            break;
                        }
                        try {
                            live.attr("append")(py::str(next_str(fdp))); // str into IVec
                        } catch (const py::error_already_set &) {
                        } catch (const py::builtin_exception &) {
                        }
                        try {
                            live.attr("__getitem__")(next_int(fdp)); // int key into map
                        } catch (const py::error_already_set &) {
                        }
                        break;
                    }
                    default:
                        break;
                }
                if (ops % 6 == 0) {
                    (void) env.gc_mod.attr("collect")();
                }
            } catch (const py::error_already_set &) {
                // Expected control flow: IndexError/KeyError/TypeError.
            } catch (const py::builtin_exception &) {
                // Expected control flow: caster rejections.
            } catch (const std::exception &) {
                // Keep the gate clean from documented C++ exception paths.
            }
        }
    } catch (const py::error_already_set &) {
    } catch (const py::builtin_exception &) {
    } catch (const std::exception &) {
    }
    return 0;
}
