// Fuzz harness: pybind11 interpreter/eval/GIL/iostream surface
// (interp_gil_fuzzer.cpp)
//
// Threat model: applications embedding CPython via pybind11 feed attacker-
// influenced strings through py::eval/py::exec/py::eval_file, redirect output
// through scoped_ostream_redirect, and interleave C++ sections with
// gil_scoped_release/acquire. The fuzzed bytes choose the eval mode, the code
// string, the locals/globals entries, the redirected text, and the GIL
// bracketing sequence against a fixed interpreter fixture.
//
// Coverage targets (dark in the 2026-11-25 local baseline): eval.h (36, 0%),
// embed.h (69, 0%), gil-inl.h (37, 0%), plus iostream.h (308 lines) absent
// from the fuzz build before this harness.
//
// Design notes (pair-1 learnings applied): all one-time setup -- interpreter,
// static module handles, warm-up of every op -- runs from LLVMFuzzerInitialize
// via the FuzzEnv struct, whose member order guarantees py::object handles are
// destroyed BEFORE the interpreter finalizes (file-scope static py::object
// globals caused a teardown SEGV in the rejected run). Bounded: <= 16
// ops/iteration, strings <= 64 bytes, eval-file payloads <= 256 bytes, temp
// files under /tmp with fixed names. Per-op try/catch keeps expected Python
// errors as control flow; sanitizers remain the oracle.
//
// Input layout: op opcodes and small selectors from the END of the buffer
// (one byte each, first op = last byte); strings from the FRONT
// (ConsumeRandomLengthString(64)).

#include "pybind11/pybind11.h"
#include "pybind11/embed.h"
#include "pybind11/eval.h"
#include "pybind11/gil.h"
#include "pybind11/iostream.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include <fuzzer/FuzzedDataProvider.h>

namespace py = pybind11;

namespace pybind11_fuzz {

enum Op : uint8_t {
    kEvalExpr = 0,
    kEvalStatements,
    kEvalSingleStatement,
    kExecLocals,
    kEvalFile,
    kOstreamRedirect,
    kGilReleaseAcquire,
    kGlobalsManipulate,
    kMaxOp // sentinel
};

inline uint8_t next_op(FuzzedDataProvider &fdp) {
    return fdp.ConsumeIntegralInRange<uint8_t>(0, static_cast<uint8_t>(kMaxOp - 1));
}

inline int next_int(FuzzedDataProvider &fdp) {
    return fdp.ConsumeIntegralInRange<int>(-1000, 1000);
}

inline std::string next_str(FuzzedDataProvider &fdp) { return fdp.ConsumeRandomLengthString(64); }
inline std::string eval_file_dir() { return "/tmp"; }

} // namespace pybind11_fuzz

// Interpreter fixture and static handles. Member order guarantees py::object
// handles are released before the interpreter finalizes (see header comment).
struct FuzzEnv {
    py::scoped_interpreter guard;
    py::object sys;
    py::object gc_mod;

    FuzzEnv() : guard{}, sys(py::module_::import("sys")), gc_mod(py::module_::import("gc")) {
        warm_up();
    }

    static FuzzEnv &get() {
        static FuzzEnv env;
        return env;
    }

    void warm_up() {
        try {
            (void) py::eval<py::eval_expr>(py::str("1 + 1"));
        } catch (const py::error_already_set &) {
        }
        try {
            (void) py::eval<py::eval_statements>(py::str("wv = 2"));
        } catch (const py::error_already_set &) {
        }
        try {
            (void) py::eval<py::eval_single_statement>(py::str("wv2 = 3"));
        } catch (const py::error_already_set &) {
        }
        py::dict locs;
        locs[py::str("wv3")] = 4;
        try {
            (void) py::exec(py::str("print(wv3)"), py::globals(), locs);
        } catch (const py::error_already_set &) {
        }
        try {
            std::string path = pybind11_fuzz::eval_file_dir() + "/pzf_warmup.py";
            {
                std::ofstream f(path);
                f << "wv4 = 5\n";
            }
            py::object glo = py::globals();
            py::object lo = py::dict();
            (void) py::eval_file(path, glo, lo);
            std::remove(path.c_str());
        } catch (const py::error_already_set &) {
        }
        {
            std::ostringstream out;
            {
                py::scoped_ostream_redirect redir(out, sys.attr("stdout"));
                py::print("warmup");
            }
            (void) out.str().find("warmup");
        }
        {
            py::gil_scoped_release release;
            // Pure C++ section: no Python API calls while the GIL is released.
        }
        {
            py::gil_scoped_acquire acquire;
            (void) gc_mod.attr("collect")();
        }
        py::globals()[py::str("wv5")] = 6;
        (void) py::globals().attr("get")(py::str("wv5"));
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

    try {
        int ops = 0;
        while (fdp.remaining_bytes() > 0 && ops < 16) {
            ++ops;
            const uint8_t op = next_op(fdp);
            try {
                switch (op) {
                    case kEvalExpr: {
                        std::string code = next_str(fdp);
                        py::object glo = py::globals();
                        py::object lo = py::dict();
                        (void) py::eval<py::eval_expr>(py::str(code), glo, lo);
                        break;
                    }
                    case kEvalStatements: {
                        std::string code = next_str(fdp);
                        py::object glo = py::globals();
                        py::object lo = py::dict();
                        (void) py::eval<py::eval_statements>(py::str(code), glo, lo);
                        break;
                    }
                    case kEvalSingleStatement: {
                        std::string code = next_str(fdp);
                        py::object glo = py::globals();
                        py::object lo = py::dict();
                        (void) py::eval<py::eval_single_statement>(py::str(code), glo, lo);
                        break;
                    }
                    case kExecLocals: {
                        std::string code = next_str(fdp);
                        py::dict locs;
                        locs[py::str("fzv")] = next_int(fdp);
                        locs[py::str("fzs")] = py::str(next_str(fdp));
                        (void) py::exec(py::str(code), py::globals(), locs);
                        break;
                    }
                    case kEvalFile: {
                        std::string code = next_str(fdp);
                        if (code.size() > 256) {
                            code.resize(256);
                        }
                        std::string path = pybind11_fuzz::eval_file_dir() + "/pzf_fuzz.py";
                        {
                            std::ofstream f(path, std::ios::binary | std::ios::trunc);
                            f << code;
                        }
                        try {
                            py::object glo = py::globals();
                            py::object lo = py::dict();
                            (void) py::eval_file(path, glo, lo);
                        } catch (const py::error_already_set &) {
                        }
                        std::remove(path.c_str());
                        break;
                    }
                    case kOstreamRedirect: {
                        std::string text = next_str(fdp);
                        std::ostringstream out;
                        {
                            py::scoped_ostream_redirect redir(out, env.sys.attr("stdout"));
                            py::print(text,
                                      py::arg("sep") = " ",
                                      py::arg("end") = "",
                                      py::arg("flush") = true);
                        }
                        (void) out.str().find(text);
                        break;
                    }
                    case kGilReleaseAcquire: {
                        {
                            py::gil_scoped_release release;
                            // Pure C++ section: no Python API calls while the GIL is
                            // released (single-threaded harness; exercises the
                            // release/save-restore path).
                        }
                        {
                            py::gil_scoped_acquire acquire;
                            (void) env.gc_mod.attr("collect")();
                        }
                        break;
                    }
                    case kGlobalsManipulate: {
                        std::string key = next_str(fdp);
                        int v = next_int(fdp);
                        py::object glo = py::globals();
                        glo[py::str(key)] = v;
                        (void) glo.attr("get")(py::str(key));
                        (void) glo.attr("get")(py::str("definitely-not-set"));
                        break;
                    }
                    default:
                        break;
                }
            } catch (const py::error_already_set &) {
                // Expected control flow: fuzzed code fails to parse/run.
            } catch (const py::builtin_exception &) {
                // Expected control flow: pybind11 eval/redirect rejections.
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
