#pragma once

#include "common.h"
#include "python_compat.h"
#include "structmember.h"
#include "vendor/optional.h"

#include <array>
#include <iostream>

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

// Forward declarations
struct pybind_function;
template <typename F>
inline PyObject *handle_exception(const F &f);
inline PyObject *raise_type_error(pybind_function **functions,
                                  size_t num_functions,
                                  PyObject *const *args,
                                  size_t nargs,
                                  PyObject *kwnames);
struct function_overload_set;
// End of forward declarations

// We define a Python type, pybind_function, which is used for all pybind functions
// pybind_function is a wrapper around a Functor class.
//
// We implement 3 different Functors in function_record.h that are used with this class.

// Get the PyTypeObject for pybind_functions. This is stored in local_internals, so unique per
// extension.
inline PyTypeObject *pybind_function_type();

// The core of the pybind_function class, the callback pointer
// Each pybind_function contains one of these
using pybind_function_ptr
    = tl::optional<PyObject *> (*)(pybind_function &, PyObject *const *, size_t, PyObject *, bool);

// The definition of the PyObject pybind_function
struct pybind_function {
    PyObject_VAR_HEAD

        // We construct a pybind_function for a target Functor by taking in the arguments directly
        // This lets us emplace the Functor directly within the pybind_function's storage
        template <typename Functor, typename... Args>
        explicit pybind_function(Functor *, Args &&...args) {
        using TypeToStore = typename std::remove_reference<Functor>::type;

        // Construct the target Functor within the pybind_function
        // Note the alignment requirements
        {
            size_t size = item_size<TypeToStore>();
            data = (char *) (this) + sizeof(pybind_function);

            bool align_result = align(alignof(TypeToStore), sizeof(TypeToStore), data, size);
            assert(align_result);

            if (!align_result) {
                pybind11_fail("Internal error in pybind11, should never happen");
            }

            new (data) TypeToStore(std::forward<Args>(args)...);
        }

        auto *ptr = reinterpret_cast<TypeToStore *>(data);

        deleter = [](void *f) {
            auto *item = reinterpret_cast<TypeToStore *>(f);
            item->~TypeToStore();
        };

        // The core function that applies the Functor
        func = [](pybind_function &func,
                  PyObject *const *args,
                  size_t nargs_with_flag,
                  PyObject *kwnames,
                  bool force_noconvert) -> tl::optional<PyObject *> {
            auto *functor = reinterpret_cast<TypeToStore *>(func.data);
            size_t nargs = pybind_vectorcall_nargs(nargs_with_flag);
            return (*functor)(args, nargs, kwnames, force_noconvert);
        };

        // A vectorcall implementation for that Functor
        // Same as above, but with exception handling
        vectorcall = [](PyObject *self_ptr,
                        PyObject *const *args,
                        size_t nargs_with_flag,
                        PyObject *kwnames) -> PyObject * {
            assert(PyObject_TypeCheck(self_ptr, pybind_function_type()));

            return handle_exception([&]() -> PyObject * {
                pybind_function *func = (pybind_function *) self_ptr;
                std::string n = func->name;
                TypeToStore *functor = (TypeToStore *) func->data;
                size_t nargs = pybind_vectorcall_nargs(nargs_with_flag);
                auto result = (*functor)(args, nargs, kwnames, false);

                if (!result) {
                    return raise_type_error(&func, 1, args, nargs, kwnames);
                }

                return *result;
            });
        };

        // Setup the metadata information
        {
            name = ptr->name.c_str();

            if (ptr->doc.empty()) {
                doc = nullptr;
            } else {
                doc = ptr->doc.c_str();
            }
            if (ptr->original_doc.empty()) {
                original_doc = nullptr;
            } else {
                original_doc = ptr->original_doc.c_str();
            }

            module = ptr->module.c_str();
            signature = ptr->signature.c_str();
            current_scope = ptr->current_scope;

            has_operator = ptr->has_operator;
            is_constructor = ptr->is_constructor;
            is_overload_set = std::is_same<TypeToStore, function_overload_set>::value;
        }
    }

    pybind_function(const pybind_function &other) = delete;

    ~pybind_function() { deleter(data); }

    // How much storage do we need to store a particular Functor?
    template <typename Functor>
    static constexpr size_t item_size() {
        using TypeToStore = typename std::remove_reference<Functor>::type;
        return sizeof(TypeToStore) + alignof(TypeToStore) - 1;
    }

    // Pointer to the current Functor
    void *data;
    void (*deleter)(void *);

    // Primary function pointers
    pybind_function_ptr func;
    pybind_vectorcallfunc vectorcall;

    // Metadata
    bool is_overload_set;
    bool has_operator;
    bool is_constructor;

    const char *name;
    const char *doc;
    const char *original_doc;
    const char *module;
    const char *signature;
    void *current_scope;
};

static_assert(std::is_standard_layout<pybind_function>::value,
              "pybind_function must be standard layout for offsetof");

// The data necessary to store the PyTypeObject object for pybind_function
struct pybind_function_type_data {
    PyTypeObject type;
    std::array<PyMemberDef, 4> members;

    pybind_function_type_data() {

        PYBIND11_WARNING_PUSH
        PYBIND11_WARNING_DISABLE_GCC("-Wmissing-field-initializers")

        type = {PyVarObject_HEAD_INIT(&PyType_Type, 0)};
        PyMemberDef name_def = {const_cast<char *>("__name__"),
                                T_STRING,
                                static_cast<Py_ssize_t>(offsetof(pybind_function, name))};
        PyMemberDef doc_def = {const_cast<char *>("__doc__"),
                               T_STRING,
                               static_cast<Py_ssize_t>(offsetof(pybind_function, doc))};
        PyMemberDef module_def = {const_cast<char *>("__module__"),
                                  T_STRING,
                                  static_cast<Py_ssize_t>(offsetof(pybind_function, module))};
        PyMemberDef null_def = {nullptr};

        members[0] = name_def;
        members[1] = doc_def;
        members[2] = module_def;
        members[3] = null_def;

        PYBIND11_WARNING_POP

        type.tp_getattro = PyObject_GenericGetAttr;
        type.tp_name = "pybind11_function";
        type.tp_basicsize = sizeof(pybind_function);
        type.tp_itemsize = 1;
        type.tp_dealloc = [](PyObject *self) {
            pybind_function *actual = (pybind_function *) self;
            actual->~pybind_function();

            Py_TYPE(self)->tp_free((PyObject *) self);
        };
        type.tp_members = members.data();
        type.tp_call = pybind_vectorcall_call;

        set_vectorcall_set_offset(type, offsetof(pybind_function, vectorcall));

        type.tp_flags = Py_TPFLAGS_DEFAULT | pybind_vectorcall_flag;

#ifdef Py_TPFLAGS_METHOD_DESCRIPTOR
        type.tp_flags |= Py_TPFLAGS_METHOD_DESCRIPTOR;
#endif

        type.tp_descr_get = [](PyObject *self, PyObject *obj, PyObject *) -> PyObject * {
            if (obj == nullptr) {
                Py_INCREF(self);
                return self;
            } else {
                PyObject *result_method = PyMethod_New(self, obj);
                if (result_method == nullptr) {
                    pybind11_fail("Could not create method binder");
                }
                return result_method;
            }
        };

        int ret = PyType_Ready(&type);
        if (ret != 0) {
            pybind11_fail("Could not create pybind_function type");
        }
    }
};

// A backup implementation of vectorcall for older versions of Python
inline PyObject *pybind_backup_impl_vectorcall(PyObject *callable,
                                               PyObject *const *args,
                                               size_t nargs,
                                               PyObject *kwnames) {
    assert(PyObject_TypeCheck(callable, pybind_function_type()));

    pybind_function *func = (pybind_function *) callable;
    return func->vectorcall(callable, args, nargs, kwnames);
}

// Create a new pybind_function for a given functor
// Note that this takes the arguments for the constructor and
// emplaces it within the pybind_function
template <typename Functor, typename... Args>
handle create_pybind_function(Args &&...args) {
    pybind_function *om = PyObject_NewVar(
        pybind_function, pybind_function_type(), pybind_function::item_size<Functor>());

    new (om) pybind_function((Functor *) nullptr, std::forward<Args>(args)...);

    return (PyObject *) om;
}

inline void append_note_if_missing_header_is_suspected(std::string &msg) {
    if (msg.find("std::") != std::string::npos) {
        msg += "\n\n"
               "Did you forget to `#include <pybind11/stl.h>`? Or <pybind11/complex.h>,\n"
               "<pybind11/functional.h>, <pybind11/chrono.h>, etc. Some automatic\n"
               "conversions are optional and require extra headers to be included\n"
               "when compiling your pybind11 module.";
    }
}

inline PyObject *raise_type_error(pybind_function **functions,
                                  size_t num_functions,
                                  PyObject *const *args,
                                  size_t nargs,
                                  PyObject *kwnames) {
    pybind_function *first_function = functions[0];

    if (first_function->has_operator) {
        return handle(Py_NotImplemented).inc_ref().ptr();
    }

    bool is_constructor = first_function->is_constructor;
    std::string msg;

    bool invalid_constructor
        = is_constructor
          && (nargs == 0
              || !PyObject_TypeCheck(args[0], (PyTypeObject *) first_function->current_scope));

    if (invalid_constructor) {
        msg = "__init__(self, ...) called with invalid or missing `self` argument";
    } else {
        msg = std::string(first_function->name) + "(): incompatible "
              + std::string(is_constructor ? "constructor" : "function")
              + " arguments. The following argument types are supported:\n";

        int ctr = 0;
        for (size_t i = 0; i < num_functions; i++) {
            pybind_function &record = *functions[i];
            msg += "    " + std::to_string(++ctr) + ". ";

            bool wrote_sig = false;
            if (is_constructor) {
                // For a constructor, rewrite `(self: Object, arg0, ...) -> NoneType` as
                // `Object(arg0, ...)`
                std::string sig = record.signature;
                size_t start = sig.find('(') + 7; // skip "(self: "
                if (start < sig.size()) {
                    // End at the , for the next argument
                    size_t end = sig.find(", "), next = end + 2;
                    size_t ret = sig.rfind(" -> ");
                    // Or the ), if there is no comma:
                    if (end >= sig.size()) {
                        next = end = sig.find(')');
                    }
                    if (start < end && next < sig.size()) {
                        msg.append(sig, start, end - start);
                        msg += '(';
                        msg.append(sig, next, ret - next);
                        wrote_sig = true;
                    }
                }
            }
            if (!wrote_sig) {
                msg += record.signature;
            }

            msg += '\n';
        }
        msg += "\nInvoked with: ";
        bool some_args = false;
        for (size_t ti = is_constructor ? 1 : 0; ti < nargs; ++ti) {
            if (!some_args) {
                some_args = true;
            } else {
                msg += ", ";
            }
            try {
                msg += pybind11::repr(args[ti]);
            } catch (const error_already_set &) {
                msg += "<repr raised Error>";
            }
        }
        if (kwnames && PyTuple_GET_SIZE(kwnames) > 0) {
            if (some_args) {
                msg += "; ";
            }
            msg += "kwargs: ";
            bool first = true;
            for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(kwnames); i++) {
                PyObject *entry = PyTuple_GET_ITEM(kwnames, i);
                if (first) {
                    first = false;
                } else {
                    msg += ", ";
                }
                msg += PyUnicode_AsUTF8(entry);
                msg += "=";
                try {
                    msg += pybind11::repr(args[nargs + static_cast<size_t>(i)]);
                } catch (const error_already_set &) {
                    msg += "<repr raised Error>";
                }
            }
        }

        append_note_if_missing_header_is_suspected(msg);
    }

    if (PyErr_Occurred()) {
        // #HelpAppreciated: unit test coverage for this branch.
        raise_from(PyExc_TypeError, msg.c_str());
    } else {
        PyErr_SetString(PyExc_TypeError, msg.c_str());
    }
    return nullptr;
}

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
