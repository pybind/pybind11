#pragma once

#include "common.h"
#include "internal_pytypes.h"
#include "python_compat.h"
#include "vendor/optional.h"

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

template <typename... Args>
struct process_attributes;

template <typename T>
using is_call_guard = is_instantiation<call_guard, T>;

/// Extract the ``type`` from the first `call_guard` in `Extras...` (or `void_type` if none
/// found)
template <typename... Extra>
using extract_guard_t = typename exactly_one_t<is_call_guard, call_guard<>, Extra...>::type;

// A version of min that is usable in constexpr contexts
template <typename T>
constexpr T constexpr_min(T a, T b) {
    return a > b ? b : a;
}

/// Check the number of named arguments at compile time
template <typename... Extra,
          size_t named = constexpr_sum(std::is_base_of<arg, Extra>::value...),
          size_t self = constexpr_sum(std::is_same<is_method, Extra>::value...)>
constexpr bool expected_num_args(size_t nargs, bool has_args, bool has_kwargs) {
    PYBIND11_WORKAROUND_INCORRECT_MSVC_C4100(nargs, has_args, has_kwargs);
    return named == 0 || (self + named + size_t(has_args) + size_t(has_kwargs)) == nargs;
}

/// Internal data structure which holds metadata about a keyword argument
struct argument_record {
    std::string name;
    std::string desc;
    object value;     ///< Associated Python object
    bool convert : 1; ///< True if the argument is allowed to convert when loading
    bool none : 1;    ///< True if None is allowed when loading

    argument_record() : name(""), desc(""), value(), convert(true), none(true) {}
};

template <typename...>
struct functor_metadata {};

template <typename Return, typename... Args, typename... Extra>
struct functor_metadata<Return, std::tuple<Args...>, std::tuple<Extra...>> {
    using cast_in = argument_loader<Args...>;
    using cast_out = make_caster<conditional_t<std::is_void<Return>::value, void_type, Return>>;
    static constexpr const size_t nargs = sizeof...(Args);

    static constexpr const bool has_args = cast_in::has_args;
    static constexpr const size_t args_pos = static_cast<size_t>(cast_in::args_pos);

    static constexpr const bool has_kwargs = cast_in::has_kwargs;

    static constexpr const bool has_kw_only_args
        = any_of<std::is_same<kw_only, Extra>...>::value,
        has_pos_only_args = any_of<std::is_same<pos_only, Extra>...>::value,
        has_arg_annotations = any_of<is_keyword<Extra>...>::value,
        new_style_constructor = any_of<std::is_same<is_new_style_constructor, Extra>...>::value,
        is_prepend = any_of<std::is_same<prepend, Extra>...>::value,
        is_temporary_casts = !any_of<std::is_same<has_no_temporary_casts, Extra>...>::value,
        has_operator = any_of<std::is_same<is_operator, Extra>...>::value;

    static constexpr const bool has_self = any_of<std::is_same<is_method, Extra>...>::value;

    static constexpr const size_t start_index = has_self ? 1 : 0;

    using IndexedExtra =
        typename create_indexed_sequence<is_keyword, start_index, args_pos, Extra..., void>::value;

    static constexpr size_t kw_only_pos_offset = constexpr_min(
        (size_t) constexpr_first_tuple<is_kw_only, IndexedExtra>::value, sizeof...(Extra));
    static constexpr size_t kw_only_pos
        = std::tuple_element<kw_only_pos_offset, IndexedExtra>::type::index;

    static constexpr size_t pos_only_pos_offset = constexpr_min(
        (size_t) constexpr_first_tuple<is_pos_only, IndexedExtra>::value, sizeof...(Extra));
    static constexpr size_t pos_only_pos
        = std::tuple_element<pos_only_pos_offset, IndexedExtra>::type::index;

    static constexpr const size_t nargs_pos
        = has_kw_only_args ? (kw_only_pos - static_cast<size_t>(has_args))
                           : (has_args ? args_pos : (has_kwargs ? nargs - 1 : nargs));

    static constexpr size_t nargs_pos_only = has_pos_only_args ? pos_only_pos : start_index;

    static_assert(
        expected_num_args<Extra...>(sizeof...(Args), has_args, has_kwargs),
        "The number of argument annotations does not match the number of function arguments");

    static_assert(has_arg_annotations || !has_kw_only_args,
                  "py::kw_only requires the use of argument annotations");
    static_assert(has_arg_annotations || !has_pos_only_args,
                  "py::pos_only requires the use of argument annotations (for docstrings "
                  "and aligning the annotations to the argument)");

    static_assert(constexpr_sum(is_kw_only<Extra>::value...) <= 1,
                  "py::kw_only may be specified only once");
    static_assert(constexpr_sum(is_pos_only<Extra>::value...) <= 1,
                  "py::pos_only may be specified only once");
    static_assert(!(has_kw_only_args && has_pos_only_args) || pos_only_pos < kw_only_pos,
                  "py::pos_only must come before py::kw_only");
    static_assert(!(has_args && has_kw_only_args) || (kw_only_pos == args_pos + 1),
                  "py::kw_only must come before the args parameter");

    static_assert(!(new_style_constructor) || (args_pos >= 1),
                  "A constructor must have at least one argument, the value and holder");

    std::array<argument_record, nargs> argument_info;
    size_t arginfo_index{0};

    // TODO: These std::strings are unnecessary allocations
    std::unordered_map<std::string, ssize_t> argument_index;

    /// Return value policy associated with this function
    return_value_policy policy = return_value_policy::automatic;

    std::string doc;
    std::string original_doc;

    std::string name;
    std::string module;
    std::string signature;

    /// Python handle to the parent current_scope (a class or a module)
    handle current_scope;

    explicit functor_metadata(const Extra &...extra) {
        process_attributes<Extra...>::init(extra..., *this);
        original_doc = doc;
        doc = "";
        init_signature_and_doc();

        if (name == "__init__" && !new_style_constructor) {
            pybind11_fail("Old style constructors are no longer supported.");
        }

        while (arginfo_index < argument_info.size()) {
            auto &args = argument_info[arginfo_index++];
            args.value = none();
        }

        if (current_scope) {
            if (hasattr(current_scope, "__module__")) {
                module = current_scope.attr("__module__").template cast<std::string>();
            } else if (hasattr(current_scope, "__name__")) {
                module = current_scope.attr("__name__").template cast<std::string>();
            }
        }
    }

    functor_metadata(const functor_metadata &) = delete;

    void init_signature_and_doc() {
        constexpr auto sig
            = const_name("(") + cast_in::arg_names + const_name(") -> ") + cast_out::name;
        PYBIND11_DESCR_CONSTEXPR auto types = decltype(sig)::types();
        const char *text = sig.text;

        /* Generate a proper function signature */
        size_t type_index = 0, arg_index = 0;
        bool is_starred = false;
        for (const auto *pc = text; *pc != '\0'; ++pc) {
            const auto c = *pc;

            if (c == '{') {
                // Write arg name for everything except *args and **kwargs.
                is_starred = *(pc + 1) == '*';
                if (is_starred) {
                    arg_index++;
                    continue;
                }
                // Separator for keyword-only arguments, placed before the kw
                // arguments start (unless we are already putting an *args)
                if (!has_args && arg_index == nargs_pos) {
                    signature += "*, ";
                }
                if (!argument_info[arg_index].name.empty()) {
                    signature += argument_info[arg_index].name;
                } else {
                    signature += "arg" + std::to_string(arg_index - (has_self ? 1 : 0));
                }
                signature += ": ";
            } else if (c == '}') {
                // Write default value if available.
                if (!is_starred && !argument_info[arg_index].desc.empty()) {
                    signature += " = ";
                    signature += argument_info[arg_index].desc;
                }
                // Separator for positional-only arguments (placed after the
                // argument, rather than before like *
                if (has_pos_only_args && (arg_index + 1) == nargs_pos_only) {
                    signature += ", /";
                }
                if (!is_starred) {
                    arg_index++;
                }
            } else if (c == '%') {
                const std::type_info *t = types[type_index++];
                if (!t) {
                    pybind11_fail("Internal error while parsing type signature (1)");
                }
                if (auto *tinfo = detail::get_type_info(*t)) {
                    handle th((PyObject *) tinfo->type);
                    signature += th.attr("__module__").cast<std::string>() + "."
                                 + th.attr("__qualname__").cast<std::string>();
                } else if (new_style_constructor && arg_index == 0) {
                    // A new-style `__init__` takes `self` as `value_and_holder`.
                    // Rewrite it to the proper class type.
                    signature += current_scope.attr("__module__").template cast<std::string>()
                                 + "."
                                 + current_scope.attr("__qualname__").template cast<std::string>();
                } else {
                    std::string tname(t->name());
                    detail::clean_type_id(tname);
                    signature += tname;
                }
            } else {
                signature += c;
            }
        }

        if (arg_index != nargs || types[type_index] != nullptr) {
            pybind11_fail("Internal error while parsing type signature (2) "
                          + std::to_string(arg_index) + " " + std::to_string(nargs) + text);
        }

        /* Create a nice pydoc rec including all signatures and
           docstrings of the functions in the overload chain */
        // Then specific overload signatures
        if (options::show_function_signatures()) {
            doc += name;
            doc += signature;
            doc += '\n';
        }
        if (!original_doc.empty() && options::show_user_defined_docstrings()) {
            // If we're appending another docstring, and aren't printing function signatures,
            // we need to append a newline first:
            if (options::show_function_signatures()) {
                doc += '\n';
            }
            doc += original_doc;
            if (options::show_function_signatures()) {
                doc += '\n';
            }
        }
    }

    // Process the arguments
    // Fills in call_args, arg_loader, and self_value_and_holder
    //
    // Returns true if the argument parsing was successful
    bool process_args(std::array<handle, nargs> &call_args,
                      cast_in &arg_loader,
                      value_and_holder &self_value_and_holder,
                      PyObject *const *args,
                      size_t nargs_input,
                      PyObject *kwnames,
                      bool force_noconvert = false) const {
#if defined(__CUDACC__)
#    pragma push
#    pragma diag_suppress 186 //  pointless comparison of unsigned integer with zero
#endif

        for (size_t i = 0; i < nargs; i++) {
            assert(!call_args[i]);
        }

        object args_ref;
        object kwargs_ref;

        for (size_t i = 0; i < constexpr_min(nargs_input, nargs_pos); i++) {
            if (i == 0 && new_style_constructor) {
                if (!PyObject_TypeCheck(args[i], (PyTypeObject *) current_scope.ptr())) {
                    return false;
                }

                auto *const tinfo = get_type_info((PyTypeObject *) current_scope.ptr());
                auto *const pi = reinterpret_cast<instance *>(args[i]);
                self_value_and_holder = pi->get_value_and_holder(tinfo, true);

                call_args[i] = reinterpret_cast<PyObject *>(&self_value_and_holder);
            } else {
                call_args[i] = args[i];
            }
        }

        if (nargs_input > nargs_pos) {
            if (has_args) {
                args_ref = tuple(nargs_input - nargs_pos);
                for (size_t i = nargs_pos; i < nargs_input; i++) {
                    Py_XINCREF(args[i]);
                    PyTuple_SET_ITEM(args_ref.ptr(), i - nargs_pos, args[i]);
                }
            } else {
                return false;
            }
        }

        if (has_args) {
            if (!args_ref) {
                args_ref = tuple();
            }
            call_args[nargs_pos] = args_ref.ptr();
        }

        if (has_kwargs) {
            kwargs_ref = dict();
            call_args[nargs - 1] = kwargs_ref.ptr();
        }

        if (kwnames != nullptr) {
            for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(kwnames); i++) {
                PyObject *entry = PyTuple_GET_ITEM(kwnames, i);
                const char *entry_name = PyUnicode_AsUTF8(entry);
                auto iter = argument_index.find(entry_name);
                if (iter == std::end(argument_index)) {
                    if (has_kwargs) {
                        PyDict_SetItem(
                            kwargs_ref.ptr(), entry, args[static_cast<size_t>(i) + nargs_input]);
                    } else {
                        return false;
                    }
                } else {
                    ssize_t arg_index = iter->second;
                    if (arg_index < 0) {
                        return false;
                    }
                    if (call_args[static_cast<size_t>(arg_index)]) {
                        return false;
                    }
                    call_args[static_cast<size_t>(arg_index)]
                        = args[static_cast<size_t>(i) + nargs_input];
                }
            }
        }

        for (size_t i = 0; i < nargs; i++) {
            const auto &arg_rec = argument_info[i];

            if (!call_args[i]) {
                call_args[i] = arg_rec.value;
            }

            if (!arg_rec.none && call_args[i].is_none()) {
                return false;
            }
        }

        return arg_loader.load_args(call_args, argument_info, force_noconvert);
#if defined(__CUDACC__)
#    pragma pop
#endif
    }

    // Cast the return type to a handle
    template <typename T>
    handle cast_result(T &&result, return_value_policy final_policy, handle parent) const {
        return cast_out::cast(std::forward<T>(result), final_policy, parent);
    }
};

// A pybind_function compatible functor that wraps another functor.
template <typename...>
struct function_wrapper {};

template <typename Return, typename... Args, typename... Extra, typename Func>
struct function_wrapper<Return, std::tuple<Args...>, std::tuple<Extra...>, Func> {

    typename std::remove_reference<Func>::type f;
    using metadata_type = functor_metadata<Return, std::tuple<Args...>, std::tuple<Extra...>>;
    const metadata_type metadata;
    const std::string &name;
    const std::string &signature;
    const std::string &module;
    const std::string &doc;
    const std::string &original_doc;
    void *current_scope;
    const bool has_operator;
    const bool is_constructor;

    explicit function_wrapper(Func &&in_f, const Extra &...extra)
        : f(std::forward<Func>(in_f)), metadata(extra...), name(metadata.name),
          signature(metadata.signature), module(metadata.module), doc(metadata.doc),
          original_doc(metadata.original_doc), current_scope(metadata.current_scope.ptr()),
          has_operator(metadata.has_operator), is_constructor(metadata.new_style_constructor) {}

    function_wrapper(const function_wrapper &other) = delete;

    tl::optional<PyObject *>
    operator()(PyObject *const *args, size_t nargs, PyObject *kwnames, bool force_noconvert) {

        {
            tl::optional<loader_life_support> life_support;
            if (metadata.is_temporary_casts) {
                life_support.emplace();
            }

            /* Override policy for rvalues -- usually to enforce rvp::move on an rvalue */
            return_value_policy final_policy
                = return_value_policy_override<Return>::policy(metadata.policy);

            /* Function current_scope guard -- defaults to the compile-to-nothing `void_type` */
            using Guard = extract_guard_t<Extra...>;

            std::array<handle, sizeof...(Args)> call_args;

            typename metadata_type::cast_in loader;
            value_and_holder self_value_and_holder;
            bool valid_args = metadata.process_args(
                call_args, loader, self_value_and_holder, args, nargs, kwnames, force_noconvert);

            if (!valid_args) {
                return tl::nullopt;
            }

            if (metadata_type::new_style_constructor
                && self_value_and_holder.instance_registered()) {
                PyErr_SetString(PyExc_SystemError,
                                "Trying to call __init__ a second time on a C++ class, invalid");
                return nullptr;
            }

            handle parent;
            if (nargs > 0) {
                parent = args[0];
            }

            /* Invoke call policy pre-call hook */
            process_attributes<Extra...>::precall(call_args, parent);

            handle result;
            try {
                /* Perform the functioe call */
                result = metadata.cast_result(
                    std::move(loader).template call<Return, Guard>(f), final_policy, parent);
            } catch (reference_cast_error &) {
                return tl::nullopt;
            }

            if (!result) {
                std::string msg = "Unable to convert function return value to a Python type!  The "
                                  "signature was\n\t";
                msg += signature;
                append_note_if_missing_header_is_suspected(msg);

                if (PyErr_Occurred()) {
                    raise_from(PyExc_TypeError, msg.c_str());
                    return nullptr;
                }
                PyErr_SetString(PyExc_TypeError, msg.c_str());
                return nullptr;
            }

            /* Invoke call policy post-call hook */
            process_attributes<Extra...>::postcall(call_args, parent, result);

            if (metadata_type::new_style_constructor
                && !self_value_and_holder.holder_constructed()) {
                auto *pi = reinterpret_cast<instance *>(parent.ptr());
                self_value_and_holder.type->init_instance(pi, nullptr);
            }

            return result.ptr();
        }
    }
};

struct function_overload_set {
    function_overload_set(handle child) {
        add_function(child, false);
        module = std::string(get_child(0).module);
        name = std::string(get_child(0).name);
        current_scope = get_child(0).current_scope;
        has_operator = get_child(0).has_operator;
        is_constructor = get_child(0).is_constructor;
    }

    void add_function(handle new_child, bool prepend) {
        assert(PyObject_TypeCheck(new_child.ptr(), pybind_function_type()));

        size_t insert_index = 0;
        if (prepend) {
            insert_index = 0;
            children.insert(std::begin(children), reinterpret_borrow<object>(new_child));
        } else {
            insert_index = children.size();
            children.push_back(reinterpret_borrow<object>(new_child));
        }

        if (get_child(insert_index).is_overload_set) {
            pybind11_fail("pybind internal error, this should never happen");
        }

        doc = "";
        if (options::show_function_signatures()) {
            // First a generic signature
            doc += name;
            doc += "(*args, **kwargs)\n";
            doc += "Overloaded function.\n\n";
        }

        bool first_user_def = true;
        for (size_t index = 0; index < children.size(); index++) {
            const auto &child = get_child(index);
            if (options::show_function_signatures()) {
                if (index > 0) {
                    doc += '\n';
                }
                doc += std::to_string(index + 1) + ". ";
                doc += child.name;
                doc += child.signature;
                doc += '\n';
            }
            if (child.original_doc != nullptr && options::show_user_defined_docstrings()) {
                // If we're appending another docstring, and aren't printing function signatures,
                // we need to append a newline first:
                if (!options::show_function_signatures()) {
                    if (first_user_def) {
                        first_user_def = false;
                    } else {
                        doc += '\n';
                    }
                }
                if (options::show_function_signatures()) {
                    doc += '\n';
                }
                doc += child.original_doc;
                if (options::show_function_signatures()) {
                    doc += '\n';
                }
            }
        }
    }

    tl::optional<PyObject *>
    operator()(PyObject *const *args, size_t nargs, PyObject *kwnames, bool /*force_noconvert*/) {
        // Pass 1, force noconvert
        for (size_t i = 0; i < children.size(); i++) {
            pybind_function &child = get_child(i);

            tl::optional<PyObject *> result = child.func(child, args, nargs, kwnames, true);
            if (!result) {
                continue;
            }
            return *result;
        }
        // Pass 2, allow conversions
        for (size_t i = 0; i < children.size(); i++) {
            pybind_function &child = get_child(i);
            tl::optional<PyObject *> result = child.func(child, args, nargs, kwnames, false);
            if (!result) {
                continue;
            }
            return *result;
        }

        std::vector<pybind_function *> functions;
        for (size_t i = 0; i < children.size(); i++) {
            functions.push_back(&get_child(i));
        }

        // Could not get any
        return raise_type_error(functions.data(), functions.size(), args, nargs, kwnames);
    }

    pybind_function &get_child(size_t index) {
        return *(pybind_function *) (children[index].ptr());
    }

    std::string name;
    std::string signature;
    std::string doc;
    std::string module;

    // Unused
    std::string original_doc;

    bool has_operator;
    bool is_constructor;

    void *current_scope;

    std::vector<object> children;
};

template <typename Func, typename Return, typename... Args, typename... Extra>
handle create_pybind_function_wrapper(Func &&f, Return (*)(Args...), const Extra &...extra) {
    using wrapper_type = function_wrapper<Return, std::tuple<Args...>, std::tuple<Extra...>, Func>;
    return create_pybind_function<wrapper_type>(std::forward<Func>(f), extra...);
}

inline handle create_pybind_function_overload_set(handle existing_function) {
    return create_pybind_function<function_overload_set>(existing_function);
}

inline handle combine_functions(handle existing_function, handle new_function, bool prepend) {
    pybind_function *new_wrapper = (pybind_function *) new_function.ptr();

    if (!PyObject_TypeCheck(existing_function.ptr(), pybind_function_type())) {
        if (!existing_function.is_none() && new_wrapper->name[0] != '_') {
            pybind11_fail(std::string("Found an existing object when trying to defined function ")
                          + new_wrapper->name);
        }

        return new_function.ptr();
    }

    pybind_function *wrapper = (pybind_function *) existing_function.ptr();

    if (wrapper->current_scope != new_wrapper->current_scope) {
        return new_function.ptr();
    }

    if (!wrapper->is_overload_set) {
        existing_function = create_pybind_function_overload_set(existing_function);
        wrapper = (pybind_function *) existing_function.ptr();
    } else {
        existing_function.inc_ref();
    }

    function_overload_set *wrapped = (function_overload_set *) wrapper->data;
    wrapped->add_function(new_function, prepend);

    wrapper->doc = wrapped->doc.c_str();

    return existing_function;
}

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
