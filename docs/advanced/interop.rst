.. _interop:

Interoperating with foreign bindings
====================================

Functions that are bound using pybind11 3.1 or later, and calls to ``py::cast()``
in pybind11 3.1 or later, can accept and return instances of types that were
bound using other binding libraries that support the `pymetabind
<https://github.com/hudson-trading/pymetabind>`__ standard, including both
some non-pybind11 libraries such as `nanobind <https://github.com/wjakob/nanobind>`
(if built with appropriate flags) and otherwise ABI-incompatible versions of
pybind11 3.1 or later (without further customization). The README for
pymetabind contains a list of other libraries that claim such support.

There are several use cases for working with foreign bindings:

- Perhaps you have a large codebase containing a number of different pybind11
  extension modules that share types with each other, and you want to upgrade
  to a new and ABI-incompatible release of pybind11 in some fashion other than
  "upgrade every module at the same time".

- Perhaps you need to work with types provided by a third-party extension
  such as PyTorch, which uses pybind11 but not the version you prefer.

- Perhaps you'd like to port some of the especially performance-sensitive
  parts of your bindings to a faster but less featureful binding framework,
  without leaving the comfortable world of pybind11 behind entirely.

Bindings provided by another framework are referred to as "foreign" by
pybind11, as contrasted with bindings provided by pybind11 which are
"native".  Interoperability with foreign bindings from other C++
binding frameworks that support pymetabind generally works
automatically, with no configuration needed beyond loading both
extension modules into the same Python interpreter. The rest of this
section covers why it matters where a type was bound, what limitations
still exist with foreign-bound types, and how you can customize their
usage or support non-C++ frameworks.

Why are foreign-bound types different?
--------------------------------------

When you bind a function with pybind11 that has a parameter or return value
of type ``T``, or use ``py::cast()`` to convert a C++ object of type ``T``
to or from Python, pybind11 has two basic approaches it can call upon
to perform the conversion:

* The *static* approach, using a :ref:`built-in <conversion_table>`
  or :ref:`custom type caster <custom_type_caster>`. These resolve their
  conversion logic at compile time using partial template specializations,
  which offers the flexibility to handle a whole family of template classes,
  but requires a hand-written (and ``#include``\d) type caster implementation
  for every kind of type that is to be converted.

* The *dynamic* approach: for C++ types where no custom type caster has been
  defined, pybind11 will do a runtime lookup in an internal table of C++ types
  that have been bound to Python types using statements like
  ``py::class_<T>(...)``. The Python types created by such binding statements
  are special in that their Python object layout has the capacity to
  directly wrap an instance of the corresponding C++ type ``T`` (you can
  contrast this with something like a Python ``str`` which does not refer
  directly or indirectly to anything that can be pointed at by a
  ``std::string*``). The from-Python and to-Python conversions are therefore
  straightforward: pull the C++ ``T*`` out of the Python object, or create
  a Python object wrapping a C++ ``T*``, respectively. Implicit conversions,
  custom holders, return value policies, and so forth add many more details
  but don't change the basic idea.

Almost all user-defined types will typically use the dynamic approach, because
it's cumbersome to include a header file with a custom type caster in every
compilation unit where you bind a function that accepts or returns an instance
of that type. It's much easier, especially in large codebases, to simply rely
on the fact that "someone", "somewhere", has executed a ``py::class_<T>(...)``
binding statement for the type that you want to use.

By default, the internal table used by the dynamic approach is shared between
all extension modules compiled against "similar enough" versions of pybind11.

.. note::

   To be more specific: In order to share bindings in this way, two extension
   modules must use the same layout and semantics for all of the internal data
   structures that need to be examined in the course of performing from- and
   to-Python conversions for their types. Such similarity is captured by a number
   called the pybind11 "internals version", which has increased a handful of times
   in pybind11's first ten years of existence post 1.0. We say that two pybind11
   versions with the same internals version number are "ABI-compatible". The C++
   compiler and standard library and sometimes the compiler flags (debug vs
   release) must also be "similar enough" in order to achieve ABI compatibility;
   the details are beyond the scope of this document, but see
   ``pybind11/conduit/pybind11_platform_abi_id.h`` if you're doing
   something unusual.

Individual types can opt out of cross-module sharing using the
``py::module_local()`` class binding directive; if they do so, they are stored
in a separate table which is visible only to their own extension module.
(See the discussion of :ref:`module-local bindings <module_local>` for more
details.) All bindings that wind up in the same internal table are fully
interoperable, exactly as if they were compiled in the same extension module.

Bindings that aren't in one of the two tables pybind11 looks at (the shared
one for all pybind11 extension modules with the same internals version, and
the local one for ``py::module_local()`` types in the current extension module)
were effectively invisible for most of pybind11's history. Their Python objects,
or their internal ``type_info`` structures used by pybind11 to store auxiliary
information, might not be laid out in the same way that the pybind11 code in the
current extension module expects; extracting a C++ object pointer, or creating a
new Python instance to wrap one, would be impossible without a "map".

pymetabind provides that map. Each framework that supports pymetabind exports
a standardized interface that other frameworks can use to discover and work with
its bindings. pybind11 may not directly understand the layout of a Python
instance whose type was bound in another framework, but it can call into
that framework to ask it to perform from-Python and to-Python conversions for
such instances, along with some supporting logic such as keep-alives and
exception translation. The APIs exposed by pymetabind are a compromise between
the different worldviews of different binding libraries, so the support is not
quite as "native" (or as fast) as it would be if everything were using the same
pybind11 internals, but it works well enough for all common use cases
and many uncommon ones.

Semantics and limitations of foreign bindings
---------------------------------------------

A *binding* is an association between a particular C++ type and a
particular Python type, with the semantics that each instance of the
Python type is associated with a single instance of the C++ type.  It
is created by a *framework* such as (a particular ABI/internals
version of) pybind11 or nanobind. It is possible for the same C++ type
to be bound by multiple frameworks (or even multiple times as
``module_local`` in separate extension modules by the same framework),
with each producing a separate Python type and thus a separate
binding. In rare cases, there can even be multiple bindings for the
same pair of (C++ type, Python type), each produced by a different
framework; use cases for this are limited to certain obscure
approaches for extending the capabilities of an existing framework
without its knowledge.

Cross-framework inheritance is not supported: a type bound
using pybind11 must only have base classes that were bound using
ABI-compatible versions of pybind11.

A function bound using pybind11 cannot perform a from-Python conversion to
``std::unique_ptr<T>`` using a foreign binding for ``T``, because pymetabind
doesn't provide any way to ask a foreign instance to relinquish its ownership.

When using a foreign binding to convert a Python object to a C++
``std::shared_ptr<T>``, pybind11 generally cannot "see inside" the instance to
find an existing ``shared_ptr`` to share ownership with, so it will create a
new ``shared_ptr`` control block that owns a reference to the Python object.
This is usually not a problem, but does mean that
``shared_ptr::use_count()`` won't work like you expect, which also means
``weak_ptr``\s might expire sooner than you intended them to. (If ``T``
inherits ``std::enable_shared_from_this``, then pybind11 can use that
to find the existing ``shared_ptr``, and will do so instead; in that case
the behavior should match that of native bindings.)

To-Python conversion of a custom holder type ``Holder<T>`` that uses a foreign
binding for ``T`` works in a somewhat roundabout fashion: the holder is moved
into new heap-allocated storage, a non-owning Python object is created, and the
holder is destroyed upon destruction of the Python object. The opposite direction
of conversion, from a Python instance of foreign-bound ``T`` to a C++
``Holder<T>``, is not supported at all unless a new holder can be created
without an existing holder instance. (Even if the Python instance was created
using a holder, there's no way to locate it.)

As explained above, type casters (both :ref:`built-in <conversion_table>` and
:ref:`custom <custom_type_caster>`) are looked up at compile time only.
pybind11 is not able to execute type casters from a different framework;
you will need to port them to a pybind11 equivalent. pymetabind only helps
with bindings, as produced by ``py::class_`` and similar statements.

:ref:`Implicit conversion <implicit_conversions>` defined using
``py::implicitly_convertible()`` can convert *from* foreign-bound types.
Implicit conversions *to* a foreign-bound type should be registered with
its binding library, not with pybind11.

When a C++-to-foreign-Python conversion is performed in a context that does
not specify the ``return_value_policy``, the policy to use is inferred using
pybind11's rules, which may differ from the foreign framework's.

As mentioned above, it is possible for multiple foreign bindings to
exist for the same C++ type, or for a particular C++ type to have both
a native pybind11 binding and one or more foreign ones. This might
occur due to separate Python extensions each having their own need to
bind a common type, as discussed in the section on :ref:`module-local
bindings <module_local>`. In such cases, pybind11 always tries
bindings for a given C++ type ``T`` in the following order:

* the pybind11 binding for ``T`` that was declared with ``py::module_local()``
  in this extension module, if any; then

* the pybind11 binding for ``T`` that was declared without ``py::module_local()``
  in either this extension module or another ABI-compatible one (drawing no
  distinction between the two), if any; then

* the pybind11 binding for ``T`` that was declared with ``py::module_local()``
  in a different ABI-compatible extension module, only if we're doing a
  from-Python conversion whose source object was directly produced by that
  binding; then

* each known foreign binding (including pybind11 bindings that were
  declared with ``py::module_local()`` in other extension modules)
  that this pybind11 extension module was explicitly told about using
  ``py::import_foreign()``, in order from most recent
  ``import_foreign()`` call to least recent; then

* each known foreign binding (including pybind11 bindings that were declared
  with ``py::module_local()`` in other extension modules) that was implicitly
  discovered, in the order in which they were bound, without making any
  distinction between other versions of pybind11 and non-pybind11 frameworks.

When performing C++-to-Python conversion of a type for which
:ref:`automatic downcasting <inheritance>` is applicable,
the downcast occurs in the binding library that is originally performing the
conversion (e.g., the library that bound the function that's returning the
object), even if the resulting Python object will then be obtained using a
foreign binding. That means foreign frameworks returning pybind11 types might
not downcast them in the same way that pybind11 does; they might only be able to
downcast from a primary base (with no this-pointer adjustment / no multiple
inheritance), or not downcast at all.

Narrowing the use of foreign bindings
-------------------------------------

By default, all pybind11 bindings are shared with other frameworks, and all
pybind11 functions can accept and return instances of types bound by other
pymetabind-supporting frameworks' extension modules that are written in C++.
To avoid this, you can call ``py::disable_foreign_export()`` and/or
``py::disable_foreign_import()``. These have global effect at the individual
extension module level, and cannot be re-enabled after being disabled.
If you want to interoperate with an extension module that was not written in C++,
you will need individual calls to ``py::import_foreign()`` as explained below.

``py::disable_foreign_export()`` prevents types bound in this extension
module from being used by other frameworks. It modestly reduces memory usage,
and can ensure that locally bound types are kept entirely private in cases where
you know that no other extension module will need to use them. It does not
improve the performance of using the bindings (calling Python functions or
creating instances), and only impacts bindings that are created *after* the
call to ``py::disable_foreign_export()``. If you want to export only some types,
you can call ``py::disable_foreign_export()`` and then use
``py::export_to_foreign(pytype)`` for each individual Python type
object you wish to make available to other frameworks. (Use ``py::type::of<T>()``
to get the Python type object for a C++ type ``T``.) For example:

.. code-block:: cpp

   PYBIND11_MODULE(my_ext, m) {
       // We only want specified types to be exported, not all of them
       py::disable_foreign_export();

       // Doghouse will not be exported
       auto house = py::class_<Doghouse>(m, "Doghouse")
           .def(py::init<std::vector<Pet>>());

       // Pet will not be exported upon creation
       auto pet = py::class_<Pet>(m, "Pet")
           .def(py::init<std::string>())
           .def("speak", &Pet::speak);

       // But you can export it using either of the following two lines
       // (both are equivalent):
       py::export_to_foreign(pet);
       py::export_to_foreign(py::type::of<Pet>());
   }

``py::disable_foreign_import()`` prevents functions bound in this
extension module from calling into foreign frameworks for type
conversions by defalt, while still allowing type bindings defined in
this extension module to be used in functions bound by foreign
frameworks.  You can use ``py::import_foreign()``, described below, to
make individual foreign types available for interoperability even if
you have disabled importation-by-default.

.. note::
   This "import" has nothing to do with the Python ``import`` statement;
   it refers to whether pybind11 will consult the pymetabind registry if
   it doesn't find a native match for a particular C++ type.

Disabling automatic importation improves performance, and might
improve correctness in rare scenarios where you rely on a given
overload *not* being executed when passed an instance of a
foreign-bound Python type with matching C++ type. It has no effect
unless an extension module from a different binding framework is
actually loaded (either before or after the call to
``py::disable_foreign_import()``). It affects all future calls to
functions bound in this extension module (regardless of whether the
functions were already bound or not) and all future calls to
``py::cast()`` in this extension module.

Importing specific foreign bindings
-----------------------------------

``py::import_foreign<T>(pytype)`` requests that the Python type object
 *pytype*, which was produced by some other binding framework that supports
pymetabind, be consulted as a potential source of conversions to and from the
C++ type ``T``. (You can tell whether a given Python type is likely to work for
this purpose by checking whether it has a ``__pymetabind_binding__`` attribute.)
As mentioned above, the relevant bindings will usually be discovered without
this call, but the explicit import request is still useful in three scenarios:

* when working with Python types that wrap native types that were not originally
  written in C++; these can't be discovered automatically because the framework
  that produced the binding didn't know how to provide a ``std::type_info`` that
  would identify the type being bound

* when you want to prefer a particular binding for a C++ type that has multiple
  bindings, and the binding you prefer might not be the one that was bound first

* when you have disabled automatic import of foreign bindings for a module by
  using ``py::disable_foreign_import()``, but you still want to use a few
  specified foreign bindings

In general, calling ``py::import_foreign()`` will make a foreign binding available
for conversions performed by any pybind11 extension modules that are ABI-compatible
with the one in which the call is made. However, in order to allow for isolation
when it is desired, an extension module that has called
``py::disable_foreign_import()`` will only consider foreign bindings named in a
later call to ``py::import_foreign()`` in that specific module.
The first call to ``py::disable_foreign_import()`` in a particular extension
module effectively reverses the effects of any prior calls to
``py::import_foreign()`` for that extension module, but *other* modules (that
don't themselves call ``py::disable_foreign_import()``) are still able to use
the formerly imported bindings.

``py::import_foreign()`` takes an optional template argument specifying which C++
type to associate the Python type with. If the foreign type was bound using another
C++ framework, such as nanobind or a different version of pybind11, then the
template argument does not need to be provided because the C++ ``std::type_info``
structure describing the type can be found by looking at the pymetabind records.
Conversely, if the foreign type is not written in C++ or is bound by
a non-C++ framework that doesn't know about ``std::type_info``, pybind11 won't
be able to figure out what the C++ type is, and therefore needs you to specify it
via a template argument to ``py::import_foreign()``.

If you *don't* supply a template argument (for importing a C++ type), then
pybind11 will check for you that the binding you're adding was compiled using a
platform C++ ABI that is consistent with the build options for your pybind11
extension. This helps to ensure that the exporter and importer mean the same
thing when they say, for example, ``std::vector<std::string>``.
The import will throw an exception if an incompatibility is detected.

If you *do* supply a template argument (for importing a
different-language type and specifying the C++ equivalent), then pybind11
will assume that you have validated compatibility yourself. Getting it
wrong can cause crashes and other sorts of undefined behavior, so if
you're working with bindings that were created in another language, make
doubly sure you're specifying a C++ type that is fully ABI-compatible with
the one used by the foreign binding.

.. code-block:: cpp

   // --- pet.h ---
   #pragma once
   #include <string>

   struct Pet {
       std::string name;
       std::string sound;

       Pet(std::string _name, std::string _sound)
         : name(std::move(_name)), sound(std::move(_sound)) {}

       std::string speak() const { return name + " goes " + sound + "!"; }
   };

   // --- pets.cc ---
   #define NB_ENABLE_INTEROP
   #include <nanobind/nanobind.h>
   #include <nanobind/stl/string.h>
   #include "pet.h"

   NB_MODULE(pets, m) {
       auto pet = nanobind::class_<Pet>(m, "Pet")
           .def(nanobind::init<std::string, std::string>())
           .def("speak", &Pet::speak);
   }

   // --- groomer.cc ---
   #include <pybind11/pybind11.h>
   #include "pet.h"

   std::string groom(const Pet& pet) {
       return pet.name + " got a haircut";
   }

   PYBIND11_MODULE(groomer, m) {
       // This example works either with or without -DMANUAL_DEMO
   #ifdef MANUAL_DEMO
       pybind11::disable_foreign_import();

       auto pet = pybind11::module_::import_("pets").attr("Pet");

       // This could go either before or after the function binding
       // (`groom` below) that relies on it
       pybind11::import_foreign(pet);

       // If Pet were bound by a non-C++ framework, you would instead say:
       // pybind11::import_foreign<Pet>(pet);
   #endif

       m.def("groom", &groom);
   }
