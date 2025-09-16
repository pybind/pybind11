#pragma once

#include <pybind11/native_enum.h>
#include <pybind11/pybind11.h>

#include <atomic>

struct PYBIND11_EXPORT_EXCEPTION SharedExc {
    int value;
};

struct Shared {
    int value;

    enum class Enum { One = 1, Two = 2 };

    struct Stats {
        std::atomic<int> constructed{0};
        std::atomic<int> copied{0};
        std::atomic<int> moved{0};
        std::atomic<int> destroyed{0};
    };
    static Stats &stats() {
        static Stats st;
        return st;
    }

    Shared(int v = 0) : value(v) { ++stats().constructed; }
    Shared(const Shared &other) : value(other.value) { ++stats().copied; }
    Shared(Shared &&other) noexcept : value(other.value) { ++stats().moved; }
    ~Shared() { ++stats().destroyed; }

    static Shared make(int v) { return Shared{v}; }
    static std::shared_ptr<Shared> make_sp(int v) { return std::make_shared<Shared>(v); }
    static std::unique_ptr<Shared> make_up(int v) {
        return std::unique_ptr<Shared>(new Shared{v});
    }
    static Enum make_enum(int v) { return Enum(v); }

    static int check(const Shared &s) { return s.value; }
    static int check_sp(std::shared_ptr<Shared> s) { return s->value; }
    static int check_up(std::unique_ptr<Shared> s) { return s->value; }
    static int check_enum(Enum e) { return (int) e; }

    static long uses(const std::shared_ptr<Shared> &s) { return s.use_count(); }

    static pybind11::dict pull_stats() {
        pybind11::dict ret;
        auto &st = stats();
        ret["construct"] = st.constructed.exchange(0);
        ret["copy"] = st.copied.exchange(0);
        ret["move"] = st.moved.exchange(0);
        ret["destroy"] = st.destroyed.exchange(0);
        return ret;
    }

    template <bool SmartHolder>
    static void bind_funcs(pybind11::module_ m) {
        m.def("make", &make);
        m.def("make_sp", &make_sp);
        if (SmartHolder) {
            m.def("make_up", &make_up);
        } else {
            // non-smart holder can't bind a unique_ptr return when the
            // holder type is shared_ptr
            m.def(
                "make_up",
                [](int v) { return make_up(v).release(); },
                pybind11::return_value_policy::take_ownership);
        }
        m.def("make_enum", &make_enum);
        m.def("check", &check);
        m.def("check_sp", &check_sp);
        m.def("check_up", &check_up);
        m.def("check_enum", &check_enum);
        m.def("uses", &uses);
        m.def("pull_stats", &pull_stats);

        m.def("export_all", []() { pybind11::interoperate_by_default(true, false); });
        m.def("import_all", []() { pybind11::interoperate_by_default(false, true); });
        m.def("export_for_interop", &pybind11::export_for_interop);
        m.def("import_for_interop", &pybind11::import_for_interop<>);
        m.def("import_for_interop_explicit", &pybind11::import_for_interop<Shared>);
        struct Other {};
        m.def("import_for_interop_wrong_type", &pybind11::import_for_interop<Other>);
    }

    template <bool SmartHolder>
    static void bind_types(pybind11::handle scope) {
        using Holder = typename std::
            conditional<SmartHolder, pybind11::smart_holder, std::shared_ptr<Shared>>::type;
        pybind11::class_<Shared, Holder>(scope, "Shared").def_readonly("value", &Shared::value);
        pybind11::native_enum<Enum>(scope, "SharedEnum", "enum.Enum")
            .value("One", Enum::One)
            .value("Two", Enum::Two)
            .finalize();
        pybind11::delattr(scope.attr("Shared"), "_pybind11_conduit_v1_");
    }
};
