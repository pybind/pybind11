#include "pybind11_tests.h"

#include <pybind11/classh.h>

#include <memory>
#include <string>

namespace pybind11_tests {
namespace classh_wip {

struct mpty {
    std::string mtxt;
};

// clang-format off

mpty        rtrn_mpty_valu() { mpty obj{"rtrn_valu"}; return obj; }
mpty&&      rtrn_mpty_rref() { static mpty obj; obj.mtxt = "rtrn_rref"; return std::move(obj); }
mpty const& rtrn_mpty_cref() { static mpty obj; obj.mtxt = "rtrn_cref"; return obj; }
mpty&       rtrn_mpty_mref() { static mpty obj; obj.mtxt = "rtrn_mref"; return obj; }
mpty const* rtrn_mpty_cptr() { return new mpty{"rtrn_cptr"}; }
mpty*       rtrn_mpty_mptr() { return new mpty{"rtrn_mptr"}; }

std::string pass_mpty_valu(mpty obj)        { return "pass_valu:" + obj.mtxt; }
std::string pass_mpty_rref(mpty&& obj)      { return "pass_rref:" + obj.mtxt; }
std::string pass_mpty_cref(mpty const& obj) { return "pass_cref:" + obj.mtxt; }
std::string pass_mpty_mref(mpty& obj)       { return "pass_mref:" + obj.mtxt; }
std::string pass_mpty_cptr(mpty const* obj) { return "pass_cptr:" + obj->mtxt; }
std::string pass_mpty_mptr(mpty* obj)       { return "pass_mptr:" + obj->mtxt; }

std::shared_ptr<mpty>       rtrn_mpty_shmp() { return std::shared_ptr<mpty      >(new mpty{"rtrn_shmp"}); }
std::shared_ptr<mpty const> rtrn_mpty_shcp() { return std::shared_ptr<mpty const>(new mpty{"rtrn_shcp"}); }

std::string pass_mpty_shmp(std::shared_ptr<mpty>       obj) { return "pass_shmp:" + obj->mtxt; }
std::string pass_mpty_shcp(std::shared_ptr<mpty const> obj) { return "pass_shcp:" + obj->mtxt; }

std::unique_ptr<mpty>       rtrn_mpty_uqmp() { return std::unique_ptr<mpty      >(new mpty{"rtrn_uqmp"}); }
std::unique_ptr<mpty const> rtrn_mpty_uqcp() { return std::unique_ptr<mpty const>(new mpty{"rtrn_uqcp"}); }

std::string pass_mpty_uqmp(std::unique_ptr<mpty      > obj) { return "pass_uqmp:" + obj->mtxt; }
std::string pass_mpty_uqcp(std::unique_ptr<mpty const> obj) { return "pass_uqcp:" + obj->mtxt; }

// clang-format on

std::string get_mtxt(mpty const &obj) { return obj.mtxt; }

} // namespace classh_wip
} // namespace pybind11_tests

namespace pybind11 {
namespace detail {

using namespace pybind11_tests::classh_wip;

template <typename T>
struct smart_holder_type_caster_load {
    bool load(handle src, bool /*convert*/) {
        if (!isinstance<T>(src))
            return false;
        auto inst  = reinterpret_cast<instance *>(src.ptr());
        auto v_h   = inst->get_value_and_holder(get_type_info(typeid(T)));
        smhldr_ptr = &v_h.holder<pybindit::memory::smart_holder>();
        return true;
    }

protected:
    pybindit::memory::smart_holder *smhldr_ptr = nullptr;
};

template <>
struct type_caster<mpty> : smart_holder_type_caster_load<mpty> {
    static constexpr auto name = _<mpty>();

    // static handle cast(mpty, ...)
    // is redundant (leads to ambiguous overloads).

    static handle cast(mpty &&src, return_value_policy /*policy*/, handle parent) {
        // type_caster_base BEGIN
        // clang-format off
        return cast(&src, return_value_policy::move, parent);
        // clang-format on
        // type_caster_base END
    }

    static handle cast(mpty const &src, return_value_policy policy, handle parent) {
        // type_caster_base BEGIN
        // clang-format off
        if (policy == return_value_policy::automatic || policy == return_value_policy::automatic_reference)
            policy = return_value_policy::copy;
        return cast(&src, policy, parent);
        // clang-format on
        // type_caster_base END
    }

    static handle cast(mpty &src, return_value_policy policy, handle parent) {
        return cast(const_cast<mpty const &>(src), policy, parent); // Mutbl2Const
    }

    static handle cast(mpty const *src, return_value_policy policy, handle parent) {
        // type_caster_base BEGIN
        // clang-format off
        auto st = src_and_type(src);
        return cast( // Originally type_caster_generic::cast.
            st.first, policy, parent, st.second,
            make_copy_constructor(src), make_move_constructor(src));
        // clang-format on
        // type_caster_base END
    }

    static handle cast(mpty *src, return_value_policy policy, handle parent) {
        return cast(const_cast<mpty const *>(src), policy, parent); // Mutbl2Const
    }

    template <typename T_>
    using cast_op_type = conditional_t<
        std::is_same<remove_reference_t<T_>, mpty const *>::value,
        mpty const *,
        conditional_t<
            std::is_same<remove_reference_t<T_>, mpty *>::value,
            mpty *,
            conditional_t<
                std::is_same<T_, mpty const &>::value,
                mpty const &,
                conditional_t<std::is_same<T_, mpty &>::value,
                              mpty &,
                              conditional_t<std::is_same<T_, mpty &&>::value, mpty &&, mpty>>>>>;

    // clang-format off

    operator mpty()        { return smhldr_ptr->lvalue_ref<mpty>(); }
    operator mpty&&() &&   { return smhldr_ptr->rvalue_ref<mpty>(); }
    operator mpty const&() { return smhldr_ptr->lvalue_ref<mpty>(); }
    operator mpty&()       { return smhldr_ptr->lvalue_ref<mpty>(); }
    operator mpty const*() { return smhldr_ptr->as_raw_ptr_unowned<mpty>(); }
    operator mpty*()       { return smhldr_ptr->as_raw_ptr_unowned<mpty>(); }

    // clang-format on

    using itype = mpty;

    // type_caster_base BEGIN
    // clang-format off

    // Returns a (pointer, type_info) pair taking care of necessary type lookup for a
    // polymorphic type (using RTTI by default, but can be overridden by specializing
    // polymorphic_type_hook). If the instance isn't derived, returns the base version.
    static std::pair<const void *, const type_info *> src_and_type(const itype *src) {
        auto &cast_type = typeid(itype);
        const std::type_info *instance_type = nullptr;
        const void *vsrc = polymorphic_type_hook<itype>::get(src, instance_type);
        if (instance_type && !same_type(cast_type, *instance_type)) {
            // This is a base pointer to a derived type. If the derived type is registered
            // with pybind11, we want to make the full derived object available.
            // In the typical case where itype is polymorphic, we get the correct
            // derived pointer (which may be != base pointer) by a dynamic_cast to
            // most derived type. If itype is not polymorphic, we won't get here
            // except via a user-provided specialization of polymorphic_type_hook,
            // and the user has promised that no this-pointer adjustment is
            // required in that case, so it's OK to use static_cast.
            if (const auto *tpi = get_type_info(*instance_type))
                return {vsrc, tpi};
        }
        // Otherwise we have either a nullptr, an `itype` pointer, or an unknown derived pointer, so
        // don't do a cast
        return type_caster_generic::src_and_type(src, cast_type, instance_type);
    }

    using Constructor = void *(*)(const void *);

    /* Only enabled when the types are {copy,move}-constructible *and* when the type
       does not have a private operator new implementation. */
    template <typename T, typename = enable_if_t<is_copy_constructible<T>::value>>
    static auto make_copy_constructor(const T *x) -> decltype(new T(*x), Constructor{}) {
        return [](const void *arg) -> void * {
            return new T(*reinterpret_cast<const T *>(arg));
        };
    }

    template <typename T, typename = enable_if_t<std::is_move_constructible<T>::value>>
    static auto make_move_constructor(const T *x) -> decltype(new T(std::move(*const_cast<T *>(x))), Constructor{}) {
        return [](const void *arg) -> void * {
            return new T(std::move(*const_cast<T *>(reinterpret_cast<const T *>(arg))));
        };
    }

    static Constructor make_copy_constructor(...) { return nullptr; }
    static Constructor make_move_constructor(...) { return nullptr; }

    // clang-format on
    // type_caster_base END

    // Originally type_caster_generic::cast.
    PYBIND11_NOINLINE static handle cast(const void *_src,
                                         return_value_policy policy,
                                         handle parent,
                                         const detail::type_info *tinfo,
                                         void *(*copy_constructor)(const void *),
                                         void *(*move_constructor)(const void *),
                                         const void *existing_holder = nullptr) {
        if (!tinfo) // no type info: error will be set already
            return handle();

        void *src = const_cast<void *>(_src);
        if (src == nullptr)
            return none().release();

        auto it_instances = get_internals().registered_instances.equal_range(src);
        for (auto it_i = it_instances.first; it_i != it_instances.second; ++it_i) {
            for (auto instance_type : detail::all_type_info(Py_TYPE(it_i->second))) {
                if (instance_type && same_type(*instance_type->cpptype, *tinfo->cpptype))
                    return handle((PyObject *) it_i->second).inc_ref();
            }
        }

        auto inst       = reinterpret_steal<object>(make_new_instance(tinfo->type));
        auto wrapper    = reinterpret_cast<instance *>(inst.ptr());
        wrapper->owned  = false;
        void *&valueptr = values_and_holders(wrapper).begin()->value_ptr();

        switch (policy) {
            case return_value_policy::automatic:
            case return_value_policy::take_ownership:
                valueptr       = src;
                wrapper->owned = true;
                break;

            case return_value_policy::automatic_reference:
            case return_value_policy::reference:
                valueptr       = src;
                wrapper->owned = false;
                break;

            case return_value_policy::copy:
                if (copy_constructor)
                    valueptr = copy_constructor(src);
                else {
#if defined(NDEBUG)
                    throw cast_error("return_value_policy = copy, but type is "
                                     "non-copyable! (compile in debug mode for details)");
#else
                    std::string type_name(tinfo->cpptype->name());
                    detail::clean_type_id(type_name);
                    throw cast_error("return_value_policy = copy, but type " + type_name
                                     + " is non-copyable!");
#endif
                }
                wrapper->owned = true;
                break;

            case return_value_policy::move:
                if (move_constructor)
                    valueptr = move_constructor(src);
                else if (copy_constructor)
                    valueptr = copy_constructor(src);
                else {
#if defined(NDEBUG)
                    throw cast_error("return_value_policy = move, but type is neither "
                                     "movable nor copyable! "
                                     "(compile in debug mode for details)");
#else
                    std::string type_name(tinfo->cpptype->name());
                    detail::clean_type_id(type_name);
                    throw cast_error("return_value_policy = move, but type " + type_name
                                     + " is neither movable nor copyable!");
#endif
                }
                wrapper->owned = true;
                break;

            case return_value_policy::reference_internal:
                valueptr       = src;
                wrapper->owned = false;
                keep_alive_impl(inst, parent);
                break;

            default:
                throw cast_error("unhandled return_value_policy: should not happen!");
        }

        tinfo->init_instance(wrapper, existing_holder);

        return inst.release();
    }
};

template <>
struct type_caster<std::shared_ptr<mpty>> : smart_holder_type_caster_load<mpty> {
    static constexpr auto name = _<std::shared_ptr<mpty>>();

    static handle
    cast(const std::shared_ptr<mpty> &src, return_value_policy policy, handle parent) {
        if (policy != return_value_policy::automatic
            && policy != return_value_policy::reference_internal) {
            // IMPROVEABLE: Error message.
            throw cast_error("Invalid return_value_policy for shared_ptr.");
        }

        auto src_raw_ptr = src.get();
        auto st          = type_caster<mpty>::src_and_type(src_raw_ptr);
        if (st.first == nullptr)
            return none().release(); // PyErr was set already.

        void *src_raw_void_ptr         = static_cast<void *>(src_raw_ptr);
        const detail::type_info *tinfo = st.second;
        auto it_instances = get_internals().registered_instances.equal_range(src_raw_void_ptr);
        // Loop copied from type_caster_generic::cast.
        for (auto it_i = it_instances.first; it_i != it_instances.second; ++it_i) {
            for (auto instance_type : detail::all_type_info(Py_TYPE(it_i->second))) {
                if (instance_type && same_type(*instance_type->cpptype, *tinfo->cpptype))
                    // MISSING: Enforcement of consistency with existing smart_holder.
                    // MISSING: keep_alive.
                    return handle((PyObject *) it_i->second).inc_ref();
            }
        }

        object inst            = reinterpret_steal<object>(make_new_instance(tinfo->type));
        instance *inst_raw_ptr = reinterpret_cast<instance *>(inst.ptr());
        inst_raw_ptr->owned    = true;
        void *&valueptr        = values_and_holders(inst_raw_ptr).begin()->value_ptr();
        valueptr               = src_raw_void_ptr;

        auto smhldr = pybindit::memory::smart_holder::from_shared_ptr(src);
        tinfo->init_instance(inst_raw_ptr, static_cast<const void *>(&smhldr));

        if (policy == return_value_policy::reference_internal)
            keep_alive_impl(inst, parent);

        return inst.release();
    }

    template <typename>
    using cast_op_type = std::shared_ptr<mpty>;

    operator std::shared_ptr<mpty>() { return smhldr_ptr->as_shared_ptr<mpty>(); }
};

template <>
struct type_caster<std::shared_ptr<mpty const>> : smart_holder_type_caster_load<mpty> {
    static constexpr auto name = _<std::shared_ptr<mpty const>>();

    static handle
    cast(const std::shared_ptr<mpty const> &src, return_value_policy policy, handle parent) {
        return type_caster<std::shared_ptr<mpty>>::cast(
            std::const_pointer_cast<mpty>(src), // Const2Mutbl
            policy,
            parent);
    }

    template <typename>
    using cast_op_type = std::shared_ptr<mpty const>;

    operator std::shared_ptr<mpty const>() { return smhldr_ptr->as_shared_ptr<mpty>(); }
};

template <>
struct type_caster<std::unique_ptr<mpty>> : smart_holder_type_caster_load<mpty> {
    static constexpr auto name = _<std::unique_ptr<mpty>>();

    static handle cast(std::unique_ptr<mpty> &&src, return_value_policy policy, handle parent) {
        if (policy != return_value_policy::automatic
            && policy != return_value_policy::reference_internal) {
            // IMPROVEABLE: Error message.
            throw cast_error("Invalid return_value_policy for unique_ptr.");
        }

        auto src_raw_ptr = src.get();
        auto st          = type_caster<mpty>::src_and_type(src_raw_ptr);
        if (st.first == nullptr)
            return none().release(); // PyErr was set already.

        void *src_raw_void_ptr         = static_cast<void *>(src_raw_ptr);
        const detail::type_info *tinfo = st.second;
        auto it_instances = get_internals().registered_instances.equal_range(src_raw_void_ptr);
        // Loop copied from type_caster_generic::cast.
        for (auto it_i = it_instances.first; it_i != it_instances.second; ++it_i) {
            for (auto instance_type : detail::all_type_info(Py_TYPE(it_i->second))) {
                if (instance_type && same_type(*instance_type->cpptype, *tinfo->cpptype))
                    throw cast_error(
                        "Invalid unique_ptr: another instance owns this pointer already.");
            }
        }

        object inst            = reinterpret_steal<object>(make_new_instance(tinfo->type));
        instance *inst_raw_ptr = reinterpret_cast<instance *>(inst.ptr());
        inst_raw_ptr->owned    = true;
        void *&valueptr        = values_and_holders(inst_raw_ptr).begin()->value_ptr();
        valueptr               = src_raw_void_ptr;

        auto smhldr = pybindit::memory::smart_holder::from_unique_ptr(std::move(src));
        tinfo->init_instance(inst_raw_ptr, static_cast<const void *>(&smhldr));

        if (policy == return_value_policy::reference_internal)
            keep_alive_impl(inst, parent);

        return inst.release();
    }

    template <typename>
    using cast_op_type = std::unique_ptr<mpty>;

    operator std::unique_ptr<mpty>() {
        // MISSING: value_and_holder value_ptr reset,  deregister_instance.
        return smhldr_ptr->as_unique_ptr<mpty>();
    }
};

template <>
struct type_caster<std::unique_ptr<mpty const>> : smart_holder_type_caster_load<mpty> {
    static constexpr auto name = _<std::unique_ptr<mpty const>>();

    static handle
    cast(std::unique_ptr<mpty const> &&src, return_value_policy policy, handle parent) {
        return type_caster<std::unique_ptr<mpty>>::cast(
            std::unique_ptr<mpty>(const_cast<mpty *>(src.release())), // Const2Mutbl
            policy,
            parent);
    }

    template <typename>
    using cast_op_type = std::unique_ptr<mpty const>;

    operator std::unique_ptr<mpty const>() {
        // MISSING: value_and_holder value_ptr reset,  deregister_instance.
        return smhldr_ptr->as_unique_ptr<mpty>();
    }
};

} // namespace detail
} // namespace pybind11

namespace pybind11_tests {
namespace classh_wip {

TEST_SUBMODULE(classh_wip, m) {
    namespace py = pybind11;

    py::classh<mpty>(m, "mpty").def(py::init<>()).def(py::init([](const std::string &mtxt) {
        mpty obj;
        obj.mtxt = mtxt;
        return obj;
    }));

    m.def("rtrn_mpty_valu", rtrn_mpty_valu);
    m.def("rtrn_mpty_rref", rtrn_mpty_rref);
    m.def("rtrn_mpty_cref", rtrn_mpty_cref);
    m.def("rtrn_mpty_mref", rtrn_mpty_mref);
    m.def("rtrn_mpty_cptr", rtrn_mpty_cptr);
    m.def("rtrn_mpty_mptr", rtrn_mpty_mptr);

    m.def("pass_mpty_valu", pass_mpty_valu);
    m.def("pass_mpty_rref", pass_mpty_rref);
    m.def("pass_mpty_cref", pass_mpty_cref);
    m.def("pass_mpty_mref", pass_mpty_mref);
    m.def("pass_mpty_cptr", pass_mpty_cptr);
    m.def("pass_mpty_mptr", pass_mpty_mptr);

    m.def("rtrn_mpty_shmp", rtrn_mpty_shmp);
    m.def("rtrn_mpty_shcp", rtrn_mpty_shcp);

    m.def("pass_mpty_shmp", pass_mpty_shmp);
    m.def("pass_mpty_shcp", pass_mpty_shcp);

    m.def("rtrn_mpty_uqmp", rtrn_mpty_uqmp);
    m.def("rtrn_mpty_uqcp", rtrn_mpty_uqcp);

    m.def("pass_mpty_uqmp", pass_mpty_uqmp);
    m.def("pass_mpty_uqcp", pass_mpty_uqcp);

    m.def("get_mtxt", get_mtxt); // Requires pass_mpty_cref to work properly.
}

} // namespace classh_wip
} // namespace pybind11_tests
