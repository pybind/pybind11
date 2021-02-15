#ifndef PYBIND11_DETAIL_INIT_H_SMART_HOLDER_INIT_INLINE_INCLUDE_SAFETY_GUARD
#error "THIS FILE MUST ONLY BE INCLUDED FROM pybind11/detail/init.h"
#endif

template <typename Class, typename D = std::default_delete<Cpp<Class>>,
          detail::enable_if_t<detail::is_smart_holder_type_caster<Cpp<Class>>::value, int> = 0>
void construct(value_and_holder &v_h, std::unique_ptr<Cpp<Class>, D> &&unq_ptr, bool need_alias) {
    auto *ptr = unq_ptr.get();
    no_nullptr(ptr);
    if (Class::has_alias && need_alias)
        throw type_error("pybind11::init(): construction failed: returned std::unique_ptr pointee "
                         "is not an alias instance");
    auto smhldr = smart_holder::from_unique_ptr(std::move(unq_ptr));
    v_h.value_ptr() = ptr;
    v_h.type->init_instance(v_h.inst, &smhldr);
}

template <typename Class, typename D = std::default_delete<Alias<Class>>,
          detail::enable_if_t<detail::is_smart_holder_type_caster<Alias<Class>>::value, int> = 0>
void construct(value_and_holder &v_h, std::unique_ptr<Alias<Class>, D> &&unq_ptr, bool /*need_alias*/) {
    auto *ptr = unq_ptr.get();
    no_nullptr(ptr);
    auto smhldr = smart_holder::from_unique_ptr(std::move(unq_ptr));
    v_h.value_ptr() = ptr;
    v_h.type->init_instance(v_h.inst, &smhldr);
}

template <typename Class,
          detail::enable_if_t<detail::is_smart_holder_type_caster<Cpp<Class>>::value, int> = 0>
void construct(value_and_holder &v_h, std::shared_ptr<Cpp<Class>> &&shd_ptr, bool need_alias) {
    auto *ptr = shd_ptr.get();
    no_nullptr(ptr);
    if (Class::has_alias && need_alias)
        throw type_error("pybind11::init(): construction failed: returned std::shared_ptr pointee "
                         "is not an alias instance");
    auto smhldr = smart_holder::from_shared_ptr(std::move(shd_ptr));
    v_h.value_ptr() = ptr;
    v_h.type->init_instance(v_h.inst, &smhldr);
}

template <typename Class,
          detail::enable_if_t<detail::is_smart_holder_type_caster<Alias<Class>>::value, int> = 0>
void construct(value_and_holder &v_h, std::shared_ptr<Alias<Class>> &&shd_ptr, bool /*need_alias*/) {
    auto *ptr = shd_ptr.get();
    no_nullptr(ptr);
    auto smhldr = smart_holder::from_shared_ptr(std::move(shd_ptr));
    v_h.value_ptr() = ptr;
    v_h.type->init_instance(v_h.inst, &smhldr);
}
