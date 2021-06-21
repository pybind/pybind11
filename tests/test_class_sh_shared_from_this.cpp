#include "pybind11_tests.h"

#include "object.h"

#include <pybind11/smart_holder.h>

#include <iostream>
#include <stdexcept>

namespace shared_from_this_custom_deleters {

template <typename T>
struct labeled_delete {
    std::string label;
    explicit labeled_delete(const std::string &label) : label{label} {}
    void operator()(T *raw_ptr) {
        std::cout << "labeled_delete::operator() " << label << std::endl;
        if (label != "SkipDelete") {
            delete raw_ptr;
        }
    }
};

struct Atype : std::enable_shared_from_this<Atype> {};

#define SHOW_USE_COUNTS                                                                           \
    std::cout << "obj1, obj2 use_counts: " << obj1.use_count() << ", " << obj2.use_count()        \
              << std::endl;

void obj1_owns() {
    std::cout << "\nobj1_owns()" << std::endl;
    std::shared_ptr<Atype> obj1(new Atype, labeled_delete<Atype>("1st"));
    std::shared_ptr<Atype> obj2(obj1.get(), labeled_delete<Atype>("SkipDelete"));
    SHOW_USE_COUNTS
    auto sft1 = obj1->shared_from_this();
    SHOW_USE_COUNTS
    auto sft2 = obj2->shared_from_this();
    SHOW_USE_COUNTS
}

void obj2_owns() {
    std::cout << "\nobj2_owns()" << std::endl;
    std::shared_ptr<Atype> obj1(new Atype, labeled_delete<Atype>("SkipDelete"));
    std::shared_ptr<Atype> obj2(obj1.get(), labeled_delete<Atype>("2nd"));
    SHOW_USE_COUNTS
    auto sft1 = obj1->shared_from_this();
    SHOW_USE_COUNTS
    auto sft2 = obj2->shared_from_this();
    SHOW_USE_COUNTS
}

void obj_sft_reset() {
    std::cout << "\nobj_sft_reset()" << std::endl;
    std::shared_ptr<Atype> obj1(new Atype, labeled_delete<Atype>("SkipDelete"));
    std::shared_ptr<Atype> obj2(obj1.get(), labeled_delete<Atype>("ThisDeletes"));
    std::shared_ptr<Atype> *obj_sft = nullptr;
    std::shared_ptr<Atype> *obj_ign = nullptr;
    {
        auto sft1 = obj1->shared_from_this();
        auto sft2 = obj2->shared_from_this();
        long uc1  = obj1.use_count();
        long uc2  = obj2.use_count();
        if (uc1 == 3 && uc2 == 1) {
            std::cout << "SHARED_FROM_THIS_REFERENT: 1" << std::endl;
            obj_sft = &obj1;
            obj_ign = &obj2;
        } else if (uc1 == 1 && uc2 == 3) {
            std::cout << "SHARED_FROM_THIS_REFERENT: 2" << std::endl;
            obj_sft = &obj2;
            obj_ign = &obj1;
        } else {
            std::cout << "SHARED_FROM_THIS_REFERENT: UNKNOWN" << std::endl;
        }
    }
    if (obj_sft == nullptr)
        throw std::runtime_error("Unexpected `use_count`s.");
    (*obj_sft).reset();
#if defined(_MSC_VER) && _MSC_VER < 1912
    std::cout << "Preempting \"Windows fatal exception: access violation\": "
                 "(*obj_ign)->shared_from_this()"
              << std::endl;
#else
    bool got_bad_weak_ptr = false;
    try {
        static_cast<void>((*obj_ign)->shared_from_this());
    } catch (const std::bad_weak_ptr &) {
        got_bad_weak_ptr = true;
    }
    std::cout << "got_bad_weak_ptr: " << got_bad_weak_ptr << std::endl;
    std::shared_ptr<Atype> obj3(obj2.get(), labeled_delete<Atype>("SkipDelete"));
    // Working again based on the shared_ptr that was created after obj_sft was reset:
    static_cast<void>((*obj_ign)->shared_from_this());
#endif
}

} // namespace shared_from_this_custom_deleters

namespace shared_ptr_reset_and_rescue_pointee_model {

struct ToBeWrapped : std::enable_shared_from_this<ToBeWrapped> {};

struct RescuingDeleter;

struct PyWrapper {
    std::unique_ptr<RescuingDeleter> rdel;
    std::shared_ptr<ToBeWrapped> wobj;
    std::shared_ptr<PyWrapper> self;
};

struct RescuingDeleter {
    PyWrapper *pyw;
    explicit RescuingDeleter(PyWrapper *pyw) : pyw{pyw} {}
    void operator()(ToBeWrapped *raw_ptr) {
        if (pyw->self != nullptr) {
#if defined(__cpp_lib_enable_shared_from_this) && (!defined(_MSC_VER) || _MSC_VER >= 1912)
            assert(raw_ptr->weak_from_this().expired()); // CRITICAL
#endif
            pyw->wobj = std::shared_ptr<ToBeWrapped>(raw_ptr, *this);
            pyw->self.reset();
        } else {
            delete raw_ptr;
        }
    }
};

std::shared_ptr<ToBeWrapped> release_to_cpp(const std::shared_ptr<PyWrapper> &pyw) {
    std::shared_ptr<ToBeWrapped> return_value = pyw->wobj;
    pyw->wobj.reset();
    pyw->self = pyw;
    return return_value;
}

void proof_of_concept() {
    std::shared_ptr<PyWrapper> pyw(new PyWrapper);
    pyw->rdel = std::unique_ptr<RescuingDeleter>(new RescuingDeleter(pyw.get()));
    pyw->wobj = std::shared_ptr<ToBeWrapped>(new ToBeWrapped, *pyw->rdel);
    std::shared_ptr<ToBeWrapped> cpp_owner = release_to_cpp(pyw);
    assert(pyw->wobj.get() == nullptr);
    assert(cpp_owner.use_count() == 1);
    {
        std::shared_ptr<ToBeWrapped> sft = cpp_owner->shared_from_this();
        assert(cpp_owner.use_count() == 2);
    }
    assert(cpp_owner.use_count() == 1);
    cpp_owner.reset();
    assert(pyw->wobj.get() != nullptr);
    assert(pyw->wobj.use_count() == 1);
    {
        std::shared_ptr<ToBeWrapped> sft = pyw->wobj->shared_from_this();
        assert(pyw->wobj.use_count() == 2);
    }
}

} // namespace shared_ptr_reset_and_rescue_pointee_model

namespace test_class_sh_shared_from_this {

// clang-format off

class MyObject3 : public std::enable_shared_from_this<MyObject3> {
public:
    MyObject3(const MyObject3 &) = default;
    MyObject3(int value) : value(value) { print_created(this, toString()); }
    std::string toString() const { return "MyObject3[" + std::to_string(value) + "]"; }
    virtual ~MyObject3() { print_destroyed(this); }
private:
    int value;
};

struct SharedFromThisRef {
    struct B : std::enable_shared_from_this<B> {
        B() { print_created(this); }
        B(const B &) : std::enable_shared_from_this<B>() { print_copy_created(this); }
        B(B &&) : std::enable_shared_from_this<B>() { print_move_created(this); }
        ~B() { print_destroyed(this); }
    };

    B value = {};
    std::shared_ptr<B> shared = std::make_shared<B>();
};

struct SharedFromThisVBase : std::enable_shared_from_this<SharedFromThisVBase> {
    SharedFromThisVBase() = default;
    SharedFromThisVBase(const SharedFromThisVBase &) = default;
    virtual ~SharedFromThisVBase() = default;
};

struct SharedFromThisVirt : virtual SharedFromThisVBase {};

// clang-format on

} // namespace test_class_sh_shared_from_this

using namespace test_class_sh_shared_from_this;

PYBIND11_SMART_HOLDER_TYPE_CASTERS(MyObject3)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(SharedFromThisRef::B)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(SharedFromThisRef)
PYBIND11_SMART_HOLDER_TYPE_CASTERS(SharedFromThisVirt)

TEST_SUBMODULE(class_sh_shared_from_this, m) {
    // clang-format off

    py::classh<MyObject3>(m, "MyObject3")
        .def(py::init<int>());
    m.def("make_myobject3_1", []() { return new MyObject3(8); });
    m.def("make_myobject3_2", []() { return std::make_shared<MyObject3>(9); });
    m.def("print_myobject3_1", [](const MyObject3 *obj) { py::print(obj->toString()); });
    m.def("print_myobject3_2", [](std::shared_ptr<MyObject3> obj) { py::print(obj->toString()); });
    m.def("print_myobject3_3", [](const std::shared_ptr<MyObject3> &obj) { py::print(obj->toString()); });
    // m.def("print_myobject3_4", [](const std::shared_ptr<MyObject3> *obj) { py::print((*obj)->toString()); });

    using B = SharedFromThisRef::B;
    py::classh<B>(m, "B");
    py::classh<SharedFromThisRef>(m, "SharedFromThisRef")
        .def(py::init<>())
        .def_readonly("bad_wp", &SharedFromThisRef::value)
        .def_property_readonly("ref", [](const SharedFromThisRef &s) -> const B & { return *s.shared; })
        .def_property_readonly("copy", [](const SharedFromThisRef &s) { return s.value; },
                               py::return_value_policy::automatic) // XXX XXX XXX copy)
        .def_readonly("holder_ref", &SharedFromThisRef::shared)
        .def_property_readonly("holder_copy", [](const SharedFromThisRef &s) { return s.shared; },
                               py::return_value_policy::automatic) // XXX XXX XXX copy)
        .def("set_ref", [](SharedFromThisRef &, const B &) { return true; })
        .def("set_holder", [](SharedFromThisRef &, std::shared_ptr<B>) { return true; });

    static std::shared_ptr<SharedFromThisVirt> sft(new SharedFromThisVirt());
    py::classh<SharedFromThisVirt>(m, "SharedFromThisVirt")
        .def_static("get", []() { return sft.get(); }, py::return_value_policy::reference);

    // clang-format on

    m.def("obj1_owns", shared_from_this_custom_deleters::obj1_owns);
    m.def("obj2_owns", shared_from_this_custom_deleters::obj2_owns);
    m.def("obj_sft_reset", shared_from_this_custom_deleters::obj_sft_reset);

    m.def("shared_ptr_reset_and_rescue_pointee_model_proof_of_concept",
          shared_ptr_reset_and_rescue_pointee_model::proof_of_concept);
}
