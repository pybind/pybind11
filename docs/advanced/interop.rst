.. _interop:

Interoperating with foreign bindings
====================================

When you bind a function with pybind11 that has a parameter of type ``T``,
its typical behavior (if ``T`` does not use a :ref:`built-in <conversion_table>`
or :ref:`custom type caster <custom_type_caster>`) is to only accept arguments
for that parameter that are Python instances of the type created by a
``py::class_<T>(...)`` binding statement, or that derive from that type,
or that match a defined :ref:`implicit conversion <implicit_conversions>`
to that type (``py::implicitly_convertible<Something, T>()``). Moreover,
if the ``py::class_<T>(...)`` binding statement was written in a different
pybind11 extension than the function that needs the ``T``, the two extensions
must be ABI-compatible: they must use similar enough versions of pybind11 that
it's safe for their respective copies of pybind11 to share their data
structures with each other.

Sometimes, you might want more flexibility than that:

- Perhaps you have a large codebase containing a number of different pybind11
  extension modules that share types with each other, and you want to upgrade
  to a new and ABI-incompatible release of pybind11 in some fashion other than
  "upgrade every module at the same time".

- Perhaps you need to work with types provided by a third-party extension
  such as PyTorch, which uses pybind11 but not the version you prefer.

- Perhaps you'd like to port some of the especially performance-sensitive
  parts of your bindings to a faster but less featureful binding framework,
  without leaving the comfortable world of pybind11 behind entirely.

To handle such situations, pybind11 can be taught to interoperate with bindings
that were not created using pybind11, or that were created with an
ABI-incompatible version of pybind11 (as long as it is new enough to support
this feature). For example, you can define a class binding for ``Pet`` in one
extension that is written using pybind11, and then write function bindings for
``void groom(Pet&)`` and ``Pet clone(const Pet&)`` in a separate extension
module that is written using `nanobind <https://nanobind.readthedocs.io/>`__, or
vice versa. The interoperability mechanism described here allows each framework
to figure out (among other things) how to get a reference to a C++ ``Pet`` out
of a Python object provided by the other framework that supposedly contains a
Pet, without knowing anything about how that framework lays out its instances.
From pybind11's perspective, nanobind and its bindings are considered "foreign".

In order for pybind11 to interoperate with another framework in this way, the
other framework must support the `pymetabind
<https://github.com/hudson-trading/pymetabind>`__ standard. See that link for
a list of frameworks that claim to do so.

Exporting pybind11 bindings for other frameworks to use
-------------------------------------------------------

In order for a type bound by pybind11 to be usable by other binding frameworks,
pybind11 must allocate a small data structure describing how others should work
with that type. While the overhead of this is low, it is not zero, so pybind11
only does so for types where you request it. Pass the Python type object to
``py::export_for_interop()``, or use ``py::interoperate_by_default()`` if you
want all types to be exported automatically as soon as they are bound.

You can use ``py::type::of<T>()`` to get the Python type object for
a C++ type. For example:

.. code-block:: cpp

   PYBIND11_MODULE(my_ext, m) {
       auto pet = py::class_<Pet>(m, "Pet")
           .def(py::init<std::string>())
           .def("speak", &Pet::speak);

       // These two lines are equivalent:
       py::export_for_interop(pet);
       py::export_for_interop(py::type::of<Pet>());
   }


Importing other frameworks' bindings for pybind11 to use
--------------------------------------------------------

In order for pybind11 to interoperate with a foreign type, the foreign framework
that bound the type must have created an interoperability record for it.
Depending on the framework, this might occur automatically or might require
an operation similar to the ``py::export_for_interop()`` described in the
previous section. (You can tell if this has happened by checking for the
presence of an attribute on the type object called ``__pymetabind_binding__``.)
Consult the other framework's documentation for details.

Once that's done, you can teach pybind11 about the foreign type by passing its
Python type object to ``py::import_for_interop()``.
This function takes an optional template argument specifying which C++ type to
associate the Python type with. If the foreign type was bound using another
C++ framework, such as nanobind or a different version of pybind11, the template
argument need not be provided because the C++ ``std::type_info`` structure
describing the type can be found by looking at the interoperability record.
On the other hand, if the foreign type is not written in C++ or is bound by
a non-C++ framework that doesn't know about ``std::type_info``, pybind11 won't
be able to figure out what the C++ type is, and needs you to specify it via
a template argument to ``py::import_for_inteorp()``.

If you *don't* supply a template argument (for importing a C++ type), then
pybind11 will check for you that the binding you're adding was compiled using a
platform C++ ABI that is consistent with the build options for your pybind11
extension. This helps to ensure that the exporter and importer mean the same
thing when they say, for example, ``std::vector<std::string>``.
The import will throw an exception if an incompatibility is detected.

If you *do* supply a template argument (for importing a
different-language type and specifying the C++ equivalent), pybind11
will assume that you have validated compatibility yourself. Getting it
wrong can cause crashes and other sorts of undefined behavior, so if
you're working with bindings that were created in another language, make
doubly sure you're specifying a C++ type that is fully ABI-compatible with
the one used by the foreign binding.

You can use ``py::interoperate_by_default()`` if you want pybind11 to
automatically import every compatible C++ type as soon as it has been
exported by another framework.

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
   #include <nanobind/nanobind.h>
   #include <nanobind/stl/string.h>
   #include "pet.h"

   NB_MODULE(pets, m) {
       auto pet = nanobind::class_<Pet>(m, "Pet")
           .def(nanobind::init<std::string, std::string>())
           .def("speak", &Pet::speak);

       nanobind::export_for_interop(pet);
   }

   // --- groomer.cc ---
   #include <pybind11/pybind11.h>
   #include "pet.h"

   std::string groom(const Pet& pet) {
       return pet.name + " got a haircut";
   }

   PYBIND11_MODULE(groomer, m) {
       auto pet = pybind11::module_::import_("pets").attr("Pet");

       // This could go either before or after the function definition that
       // relies on it
       pybind11::import_for_interop(pet);

       // If Pet were bound by a non-C++ framework, you would instead say:
       // pybind11::import_for_interop<Pet>(pet);

       m.def("groom", &groom);
   }


Automatic communication
-----------------------

In large binding projects, you might prefer to share *all* types rather than
only those you nominate. For that, pybind11 provides the
``py::interoperate_by_default()`` function. It takes two optional bool
parameters that specify whether you want automatic export and/or automatic
import; if you don't specify the parameters, then both are enabled.

Automatic export is equivalent to writing a call to ``py::export_for_interop()``
after every ``py::class_``, ``py::enum_``, or ``py::native_enum`` binding
statement in any pybind11 module that is ABI-compatible with the one in which
you wrote the call.

Automatic import is equivalent to writing a call to ``py::import_for_interop()``
after every export of a type from a different framework. It only import
bindings written in C++ with a compatible platform ABI (the same ones that
``py::import_for_interop()`` can import without a template argument);
bindings written in other languages must always be imported explicitly.

Automatic import and export apply both to types that already exist and
types that will be bound in the future. They cannot be disabled once enabled.

Here is the above example recast to use automatic communication.

.. code-block:: cpp

   // (pet.h unchanged)

   // --- pets.cc ---
   #include <nanobind/nanobind.h>
   #include <nanobind/stl/string.h>
   #include "pet.h"

   NB_MODULE(pets, m) {
       nanobind::interoperate_by_default();
       nanobind::class_<Pet>(m, "Pet")
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
       pybind11::interoperate_by_default();
       m.def("groom", &groom);
   }


Conversion semantics and caveats
--------------------------------

Cross-framework inheritance is not supported: a type bound
using pybind11 must only have base classes that were bound using
ABI-compatible versions of pybind11.

A function bound using pybind11 cannot perform a conversion to
``std::unique_ptr<T>`` using a foreign binding for ``T``, because the
interoperability mechanism doesn't provide any way to ask a foreign instance
to relinquish its ownership.

When converting from a foreign instance to ``std::shared_ptr<T>``, pybind11
generally cannot "see inside" the instance to find an existing ``shared_ptr``
to share ownership with, so it will create a new ``shared_ptr`` control block
that owns a reference to the Python object. This is usually not a problem, but
does mean that ``shared_ptr::use_count()`` won't work like you expect. (If
``T`` inherits ``std::enable_shared_from_this``, then pybind11 can use that
to find the existing ``shared_ptr``, and will do so instead.)

Type casters (both :ref:`built-in <conversion_table>` and :ref:`custom
<custom_type_caster>`) execute before the interoperability mechanism
has a chance to step in. pybind11 is not able to execute type casters from
a different framework; you will need to port them to a pybind11 equivalent.
Interoperability only helps with bindings, as produced by ``py::class_`` and
similar statements.

:ref:`Implicit conversion <implicit_conversions>` defined using
``py::implicitly_convertible()`` can convert *from* foreign types.
Implicit conversions *to* a foreign type should be registered with its
binding library, not with pybind11.

When a C++-to-foreign-Python conversion is performed in a context that does
not specify the ``return_value_policy``, the policy to use is inferred using
pybind11's rules, which may differ from the foreign framework's.

It is possible for multiple foreign bindings to exist for the same C++ type,
or for a particular C++ type to have both a native pybind11 binding
and one or more foreign ones. This might occur due to separate Python
extensions each having their own need to bind a common type, as discussed in
the section on :ref:`module-local bindings <module_local>`. In such cases,
pybind11 always tries bindings for a given C++ type ``T`` in the following order:

* the pybind11 binding for ``T`` that was declared with ``py::module_local()``
  in this extension module, if any; then

* the pybind11 binding for ``T`` that was declared without ``py::module_local()``
  in either this extension module or another ABI-compatible one (drawing no
  distinction between the two), if any; then

* if performing a from-Python conversion on an instance of a pybind11 binding
  for ``T`` that was declared with ``py::module_local()`` in a different
  but ABI-compatible module, that binding; otherwise

* each known foreign binding, in the order in which they were imported,
  without making any distinction between other versions of pybind11 and
  non-pybind11 frameworks. (If automatic import is enabled, then the import
  order will match the original export order.)

You can use the interoperability mechanism to share :ref:`module-local bindings
<module_local>` with other modules. Unlike the sharing that happens by default,
this allows you to return instances of such bindings from outside the module in
which they were defined.

When performing C++-to-Python conversion of a type for which
:ref:`automatic downcasting <inheritance>` is applicable,
the downcast occurs in the binding library that is originally performing the
conversion, even if the result will then be obtained using a foreign binding.
That means foreign frameworks returning pybind11 types might not downcast
them in the same way that pybind11 does; they might only be able to downcast
from a primary base (with no this-pointer adjustment / no multiple inheritance),
or not downcast at all.
