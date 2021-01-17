namespace pybind11 {
namespace detail {

using namespace pybind11_tests::classh_wip;

inline std::pair<bool, handle> find_existing_python_instance(void *src_void_ptr,
                                                             const detail::type_info *tinfo) {
    // Loop copied from type_caster_generic::cast.
    // IMPROVEABLE: Factor out of type_caster_generic::cast.
    auto it_instances = get_internals().registered_instances.equal_range(src_void_ptr);
    for (auto it_i = it_instances.first; it_i != it_instances.second; ++it_i) {
        for (auto instance_type : detail::all_type_info(Py_TYPE(it_i->second))) {
            if (instance_type && same_type(*instance_type->cpptype, *tinfo->cpptype))
                return std::make_pair(true, handle((PyObject *) it_i->second).inc_ref());
        }
    }
    return std::make_pair(false, handle());
}

template <typename T>
struct smart_holder_type_caster_load {
    using holder_type = pybindit::memory::smart_holder;

    bool load(handle src, bool /*convert*/) {
        if (!isinstance<T>(src))
            return false;
        auto inst  = reinterpret_cast<instance *>(src.ptr());
        loaded_v_h = inst->get_value_and_holder(get_type_info(typeid(T)));
        if (!loaded_v_h.holder_constructed()) {
            // IMPROVEABLE: Error message. A change to the existing internals is
            // needed to cleanly distinguish between uninitialized or disowned.
            throw std::runtime_error("Missing value for wrapped C++ type:"
                                     " Python instance is uninitialized or was disowned.");
        }
        loaded_smhldr_ptr = &loaded_v_h.holder<holder_type>();
        return true;
    }

    std::unique_ptr<T> loaded_as_unique_ptr() {
        void *value_void_ptr = loaded_v_h.value_ptr();
        auto unq_ptr         = loaded_smhldr_ptr->as_unique_ptr<mpty>();
        loaded_v_h.holder<holder_type>().~holder_type();
        loaded_v_h.set_holder_constructed(false);
        loaded_v_h.value_ptr() = nullptr;
        deregister_instance(loaded_v_h.inst, value_void_ptr, loaded_v_h.type);
        return unq_ptr;
    }

protected:
    value_and_holder loaded_v_h;
    holder_type *loaded_smhldr_ptr = nullptr;
};

// type_caster_base BEGIN
// clang-format off
// Helper factored out of type_caster_base.
struct make_constructor {
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
};
// clang-format on
// type_caster_base END

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
        auto st = type_caster_base<mpty>::src_and_type(src);
        return cast_const_raw_ptr( // Originally type_caster_generic::cast.
            st.first,
            policy,
            parent,
            st.second,
            make_constructor::make_copy_constructor(src),
            make_constructor::make_move_constructor(src));
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

    operator mpty()        { return loaded_smhldr_ptr->lvalue_ref<mpty>(); }
    operator mpty&&() &&   { return loaded_smhldr_ptr->rvalue_ref<mpty>(); }
    operator mpty const&() { return loaded_smhldr_ptr->lvalue_ref<mpty>(); }
    operator mpty&()       { return loaded_smhldr_ptr->lvalue_ref<mpty>(); }
    operator mpty const*() { return loaded_smhldr_ptr->as_raw_ptr_unowned<mpty>(); }
    operator mpty*()       { return loaded_smhldr_ptr->as_raw_ptr_unowned<mpty>(); }

    // clang-format on

    // Originally type_caster_generic::cast.
    PYBIND11_NOINLINE static handle cast_const_raw_ptr(const void *_src,
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

        auto existing_inst = find_existing_python_instance(src, tinfo);
        if (existing_inst.first)
            return existing_inst.second;

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
        auto st          = type_caster_base<mpty>::src_and_type(src_raw_ptr);
        if (st.first == nullptr)
            return none().release(); // PyErr was set already.

        void *src_raw_void_ptr         = static_cast<void *>(src_raw_ptr);
        const detail::type_info *tinfo = st.second;
        auto existing_inst             = find_existing_python_instance(src_raw_void_ptr, tinfo);
        if (existing_inst.first)
            // MISSING: Enforcement of consistency with existing smart_holder.
            // MISSING: keep_alive.
            return existing_inst.second;

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

    operator std::shared_ptr<mpty>() { return loaded_smhldr_ptr->as_shared_ptr<mpty>(); }
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

    operator std::shared_ptr<mpty const>() { return loaded_smhldr_ptr->as_shared_ptr<mpty>(); }
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
        auto st          = type_caster_base<mpty>::src_and_type(src_raw_ptr);
        if (st.first == nullptr)
            return none().release(); // PyErr was set already.

        void *src_raw_void_ptr         = static_cast<void *>(src_raw_ptr);
        const detail::type_info *tinfo = st.second;
        auto existing_inst             = find_existing_python_instance(src_raw_void_ptr, tinfo);
        if (existing_inst.first)
            throw cast_error("Invalid unique_ptr: another instance owns this pointer already.");

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

    operator std::unique_ptr<mpty>() { return loaded_as_unique_ptr(); }
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

    operator std::unique_ptr<mpty const>() { return loaded_as_unique_ptr(); }
};

} // namespace detail
} // namespace pybind11
