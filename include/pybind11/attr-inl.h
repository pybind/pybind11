#include "attr.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

PYBIND11_NAMESPACE_BEGIN(detail)

PYBIND11_INLINE void type_record::add_base(const std::type_info &base, void *(*caster)(void *)) {
    auto base_info = detail::get_type_info(base, false);
    if (!base_info) {
        std::string tname(base.name());
        detail::clean_type_id(tname);
        pybind11_fail("generic_type: type \"" + std::string(name) +
                        "\" referenced unknown base type \"" + tname + "\"");
    }

    if (default_holder != base_info->default_holder) {
        std::string tname(base.name());
        detail::clean_type_id(tname);
        pybind11_fail("generic_type: type \"" + std::string(name) + "\" " +
                (default_holder ? "does not have" : "has") +
                " a non-default holder type while its base \"" + tname + "\" " +
                (base_info->default_holder ? "does not" : "does"));
    }

    bases.append((PyObject *) base_info->type);

    if (base_info->type->tp_dictoffset != 0)
        dynamic_attr = true;

    if (caster)
        base_info->implicit_casts.emplace_back(type, caster);
}

PYBIND11_INLINE function_call::function_call(const function_record &f, handle p) :
        func(f), parent(p) {
    args.reserve(f.nargs);
    args_convert.reserve(f.nargs);
}

PYBIND11_INLINE void process_kwonly_arg(const arg &a, function_record *r) {
    if (!a.name || strlen(a.name) == 0)
        pybind11_fail("arg(): cannot specify an unnamed argument after an kwonly() annotation");
    ++r->nargs_kwonly;
}

PYBIND11_NAMESPACE_END(detail)

PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
