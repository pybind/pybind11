#include "pybind11/smart_holder_poc.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using pybindit::memory::smart_holder;

namespace helpers {

struct movable_int {
    int valu;
    movable_int(int v) : valu{v} {}
    movable_int(movable_int &&other) {
        valu       = other.valu;
        other.valu = 91;
    }
};

template <typename T>
struct functor_builtin_delete {
    void operator()(T *ptr) { delete ptr; }
};

template <typename T>
struct functor_other_delete : functor_builtin_delete<T> {};

} // namespace helpers

TEST_CASE("from_raw_ptr_unowned+as_raw_ptr_unowned", "[S]") {
    static int value = 19;
    auto hld         = smart_holder::from_raw_ptr_unowned(&value);
    REQUIRE(*hld.as_raw_ptr_unowned<int>() == 19);
}

TEST_CASE("from_raw_ptr_unowned+as_lvalue_ref", "[S]") {
    static int value = 19;
    auto hld         = smart_holder::from_raw_ptr_unowned(&value);
    REQUIRE(hld.as_lvalue_ref<int>() == 19);
}

TEST_CASE("from_raw_ptr_unowned+as_rvalue_ref", "[S]") {
    helpers::movable_int orig(19);
    {
        auto hld = smart_holder::from_raw_ptr_unowned(&orig);
        helpers::movable_int othr(hld.as_rvalue_ref<helpers::movable_int>());
        REQUIRE(othr.valu == 19);
        REQUIRE(orig.valu == 91);
    }
}

TEST_CASE("from_raw_ptr_unowned+as_raw_ptr_release_ownership", "[E]") {
    static int value = 19;
    auto hld         = smart_holder::from_raw_ptr_unowned(&value);
    REQUIRE_THROWS_WITH(hld.as_raw_ptr_release_ownership<int>(),
                        "Cannot disown non-owning holder (as_raw_ptr_release_ownership).");
}

TEST_CASE("from_raw_ptr_unowned+as_unique_ptr", "[E]") {
    static int value = 19;
    auto hld         = smart_holder::from_raw_ptr_unowned(&value);
    REQUIRE_THROWS_WITH(hld.as_unique_ptr<int>(),
                        "Cannot disown non-owning holder (as_unique_ptr).");
}

TEST_CASE("from_raw_ptr_unowned+as_unique_ptr_with_deleter", "[E]") {
    static int value        = 19;
    auto hld                = smart_holder::from_raw_ptr_unowned(&value);
    auto condense_for_macro = [](smart_holder &hld) {
        hld.as_unique_ptr_with_deleter<int, helpers::functor_builtin_delete<int>>();
    };
    REQUIRE_THROWS_WITH(condense_for_macro(hld),
                        "Missing unique_ptr deleter (as_unique_ptr_with_deleter).");
}

TEST_CASE("from_raw_ptr_unowned+as_shared_ptr", "[S]") {
    static int value = 19;
    auto hld         = smart_holder::from_raw_ptr_unowned(&value);
    REQUIRE(*hld.as_shared_ptr<int>() == 19);
}

TEST_CASE("from_raw_ptr_take_ownership+as_lvalue_ref", "[S]") {
    auto hld = smart_holder::from_raw_ptr_take_ownership(new int(19));
    REQUIRE(hld.has_pointee());
    REQUIRE(hld.as_lvalue_ref<int>() == 19);
}

TEST_CASE("from_raw_ptr_take_ownership+as_raw_ptr_release_ownership1", "[S]") {
    auto hld       = smart_holder::from_raw_ptr_take_ownership(new int(19));
    auto new_owner = std::unique_ptr<int>(hld.as_raw_ptr_release_ownership<int>());
    REQUIRE(!hld.has_pointee());
    REQUIRE(*new_owner == 19);
}

TEST_CASE("from_raw_ptr_take_ownership+as_raw_ptr_release_ownership2", "[E]") {
    auto hld     = smart_holder::from_raw_ptr_take_ownership(new int(19));
    auto shd_ptr = hld.as_shared_ptr<int>();
    REQUIRE_THROWS_WITH(hld.as_raw_ptr_release_ownership<int>(),
                        "Cannot disown use_count != 1 (as_raw_ptr_release_ownership).");
}

TEST_CASE("from_raw_ptr_take_ownership+as_unique_ptr1", "[S]") {
    auto hld                       = smart_holder::from_raw_ptr_take_ownership(new int(19));
    std::unique_ptr<int> new_owner = hld.as_unique_ptr<int>();
    REQUIRE(!hld.has_pointee());
    REQUIRE(*new_owner == 19);
}

TEST_CASE("from_raw_ptr_take_ownership+as_unique_ptr2", "[E]") {
    auto hld     = smart_holder::from_raw_ptr_take_ownership(new int(19));
    auto shd_ptr = hld.as_shared_ptr<int>();
    REQUIRE_THROWS_WITH(hld.as_unique_ptr<int>(), "Cannot disown use_count != 1 (as_unique_ptr).");
}

TEST_CASE("from_raw_ptr_take_ownership+as_unique_ptr_with_deleter", "[E]") {
    auto hld                = smart_holder::from_raw_ptr_take_ownership(new int(19));
    auto condense_for_macro = [](smart_holder &hld) {
        hld.as_unique_ptr_with_deleter<int, helpers::functor_builtin_delete<int>>();
    };
    REQUIRE_THROWS_WITH(condense_for_macro(hld),
                        "Missing unique_ptr deleter (as_unique_ptr_with_deleter).");
}

TEST_CASE("from_raw_ptr_take_ownership+as_shared_ptr", "[S]") {
    auto hld                       = smart_holder::from_raw_ptr_take_ownership(new int(19));
    std::shared_ptr<int> new_owner = hld.as_shared_ptr<int>();
    REQUIRE(hld.has_pointee());
    REQUIRE(*new_owner == 19);
}

TEST_CASE("from_unique_ptr+as_lvalue_ref", "[S]") {
    std::unique_ptr<int> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    REQUIRE(hld.as_lvalue_ref<int>() == 19);
}

TEST_CASE("from_unique_ptr+as_raw_ptr_release_ownership1", "[S]") {
    std::unique_ptr<int> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    auto new_owner = std::unique_ptr<int>(hld.as_raw_ptr_release_ownership<int>());
    REQUIRE(!hld.has_pointee());
    REQUIRE(*new_owner == 19);
}

TEST_CASE("from_unique_ptr+as_raw_ptr_release_ownership2", "[E]") {
    std::unique_ptr<int> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    auto shd_ptr = hld.as_shared_ptr<int>();
    REQUIRE_THROWS_WITH(hld.as_raw_ptr_release_ownership<int>(),
                        "Cannot disown use_count != 1 (as_raw_ptr_release_ownership).");
}

TEST_CASE("from_unique_ptr+as_unique_ptr1", "[S]") {
    std::unique_ptr<int> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    std::unique_ptr<int> new_owner = hld.as_unique_ptr<int>();
    REQUIRE(!hld.has_pointee());
    REQUIRE(*new_owner == 19);
}

TEST_CASE("from_unique_ptr+as_unique_ptr2", "[E]") {
    std::unique_ptr<int> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    auto shd_ptr = hld.as_shared_ptr<int>();
    REQUIRE_THROWS_WITH(hld.as_unique_ptr<int>(), "Cannot disown use_count != 1 (as_unique_ptr).");
}

TEST_CASE("from_unique_ptr+as_unique_ptr_with_deleter", "[E]") {
    std::unique_ptr<int> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    auto condense_for_macro = [](smart_holder &hld) {
        hld.as_unique_ptr_with_deleter<int, helpers::functor_builtin_delete<int>>();
    };
    REQUIRE_THROWS_WITH(condense_for_macro(hld),
                        "Missing unique_ptr deleter (as_unique_ptr_with_deleter).");
}

TEST_CASE("from_unique_ptr+as_shared_ptr", "[S]") {
    std::unique_ptr<int> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    std::shared_ptr<int> new_owner = hld.as_shared_ptr<int>();
    REQUIRE(hld.has_pointee());
    REQUIRE(*new_owner == 19);
}

TEST_CASE("from_unique_ptr_with_deleter+as_lvalue_ref", "[S]") {
    std::unique_ptr<int, helpers::functor_builtin_delete<int>> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr_with_deleter(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    REQUIRE(hld.as_lvalue_ref<int>() == 19);
}

TEST_CASE("from_unique_ptr_with_deleter+as_raw_ptr_release_ownership", "[E]") {
    std::unique_ptr<int, helpers::functor_builtin_delete<int>> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr_with_deleter(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    REQUIRE_THROWS_WITH(hld.as_raw_ptr_release_ownership<int>(),
                        "Cannot disown custom deleter (as_raw_ptr_release_ownership).");
}

TEST_CASE("from_unique_ptr_with_deleter+as_unique_ptr", "[E]") {
    std::unique_ptr<int, helpers::functor_builtin_delete<int>> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr_with_deleter(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    REQUIRE_THROWS_WITH(hld.as_unique_ptr<int>(), "Cannot disown custom deleter (as_unique_ptr).");
}

TEST_CASE("from_unique_ptr_with_deleter+as_unique_ptr_with_deleter1", "[S]") {
    std::unique_ptr<int, helpers::functor_builtin_delete<int>> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr_with_deleter(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    std::unique_ptr<int, helpers::functor_builtin_delete<int>> new_owner
        = hld.as_unique_ptr_with_deleter<int, helpers::functor_builtin_delete<int>>();
    REQUIRE(!hld.has_pointee());
    REQUIRE(*new_owner == 19);
}

TEST_CASE("from_unique_ptr_with_deleter+as_unique_ptr_with_deleter2", "[E]") {
    std::unique_ptr<int, helpers::functor_builtin_delete<int>> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr_with_deleter(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    auto condense_for_macro = [](smart_holder &hld) {
        hld.as_unique_ptr_with_deleter<int, helpers::functor_other_delete<int>>();
    };
    REQUIRE_THROWS_WITH(condense_for_macro(hld),
                        "Incompatible unique_ptr deleter (as_unique_ptr_with_deleter).");
}

TEST_CASE("from_unique_ptr_with_deleter+as_shared_ptr", "[S]") {
    std::unique_ptr<int, helpers::functor_builtin_delete<int>> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr_with_deleter(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    std::shared_ptr<int> new_owner = hld.as_shared_ptr<int>();
    REQUIRE(hld.has_pointee());
    REQUIRE(*new_owner == 19);
}

TEST_CASE("from_shared_ptr+as_lvalue_ref", "[S]") {
    std::shared_ptr<int> orig_owner(new int(19));
    auto hld = smart_holder::from_shared_ptr(orig_owner);
    REQUIRE(hld.as_lvalue_ref<int>() == 19);
}

TEST_CASE("from_shared_ptr+as_raw_ptr_release_ownership", "[E]") {
    std::shared_ptr<int> orig_owner(new int(19));
    auto hld = smart_holder::from_shared_ptr(orig_owner);
    REQUIRE_THROWS_WITH(hld.as_raw_ptr_release_ownership<int>(),
                        "Cannot disown external shared_ptr (as_raw_ptr_release_ownership).");
}

TEST_CASE("from_shared_ptr+as_unique_ptr", "[E]") {
    std::shared_ptr<int> orig_owner(new int(19));
    auto hld = smart_holder::from_shared_ptr(orig_owner);
    REQUIRE_THROWS_WITH(hld.as_unique_ptr<int>(),
                        "Cannot disown external shared_ptr (as_unique_ptr).");
}

TEST_CASE("from_shared_ptr+as_unique_ptr_with_deleter", "[E]") {
    std::shared_ptr<int> orig_owner(new int(19));
    auto hld                = smart_holder::from_shared_ptr(orig_owner);
    auto condense_for_macro = [](smart_holder &hld) {
        hld.as_unique_ptr_with_deleter<int, helpers::functor_builtin_delete<int>>();
    };
    REQUIRE_THROWS_WITH(condense_for_macro(hld),
                        "Missing unique_ptr deleter (as_unique_ptr_with_deleter).");
}

TEST_CASE("from_shared_ptr+as_shared_ptr", "[S]") {
    std::shared_ptr<int> orig_owner(new int(19));
    auto hld = smart_holder::from_shared_ptr(orig_owner);
    REQUIRE(*hld.as_shared_ptr<int>() == 19);
}

TEST_CASE("error_unpopulated_holder", "[E]") {
    smart_holder hld;
    REQUIRE_THROWS_WITH(hld.as_lvalue_ref<int>(), "Unpopulated holder (as_lvalue_ref).");
}

TEST_CASE("error_disowned_holder", "[E]") {
    auto hld = smart_holder::from_raw_ptr_take_ownership(new int(19));
    hld.as_unique_ptr<int>();
    REQUIRE_THROWS_WITH(hld.as_lvalue_ref<int>(), "Disowned holder (as_lvalue_ref).");
}

TEST_CASE("error_cannot_disown_nullptr", "[E]") {
    auto hld = smart_holder::from_raw_ptr_take_ownership(new int(19));
    hld.as_unique_ptr<int>();
    REQUIRE_THROWS_WITH(hld.as_unique_ptr<int>(), "Cannot disown nullptr (as_unique_ptr).");
}
