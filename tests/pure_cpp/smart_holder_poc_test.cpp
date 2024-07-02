#define PYBIND11_TESTS_PURE_CPP_SMART_HOLDER_POC_TEST_CPP

#include "pybind11/detail/smart_holder_poc.h"

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

// Catch uses _ internally, which breaks gettext style defines
#ifdef _
#    undef _
#endif

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using pybindit::memory::smart_holder;

namespace helpers {

struct movable_int {
    int valu;
    explicit movable_int(int v) : valu{v} {}
    movable_int(movable_int &&other) noexcept : valu(other.valu) { other.valu = 91; }
};

template <typename T>
struct functor_builtin_delete {
    void operator()(T *ptr) { delete ptr; }
#if (defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ == 8)                                   \
    || (defined(__clang_major__) && __clang_major__ == 3 && __clang_minor__ == 6)
    // Workaround for these errors:
    // gcc 4.8.5: too many initializers for 'helpers::functor_builtin_delete<int>'
    // clang 3.6: excess elements in struct initializer
    functor_builtin_delete() = default;
    functor_builtin_delete(const functor_builtin_delete &) {}
    functor_builtin_delete(functor_builtin_delete &&) {}
#endif
};

template <typename T>
struct functor_other_delete : functor_builtin_delete<T> {};

struct indestructible_int {
    int valu;
    explicit indestructible_int(int v) : valu{v} {}

private:
    ~indestructible_int() = default;
};

struct base {
    virtual int get() { return 10; }
    virtual ~base() = default;
};

struct derived : public base {
    int get() override { return 100; }
};

} // namespace helpers

TEST_CASE("from_raw_ptr_unowned+as_raw_ptr_unowned", "[S]") {
    static int value = 19;
    auto hld = smart_holder::from_raw_ptr_unowned(&value);
    REQUIRE(*hld.as_raw_ptr_unowned<int>() == 19);
}

TEST_CASE("from_raw_ptr_unowned+as_lvalue_ref", "[S]") {
    static int value = 19;
    auto hld = smart_holder::from_raw_ptr_unowned(&value);
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
    auto hld = smart_holder::from_raw_ptr_unowned(&value);
    REQUIRE_THROWS_WITH(hld.as_raw_ptr_release_ownership<int>(),
                        "Cannot disown non-owning holder (as_raw_ptr_release_ownership).");
}

TEST_CASE("from_raw_ptr_unowned+as_unique_ptr", "[E]") {
    static int value = 19;
    auto hld = smart_holder::from_raw_ptr_unowned(&value);
    REQUIRE_THROWS_WITH(hld.as_unique_ptr<int>(),
                        "Cannot disown non-owning holder (as_unique_ptr).");
}

TEST_CASE("from_raw_ptr_unowned+as_unique_ptr_with_deleter", "[E]") {
    static int value = 19;
    auto hld = smart_holder::from_raw_ptr_unowned(&value);
    REQUIRE_THROWS_WITH((hld.as_unique_ptr<int, helpers::functor_builtin_delete<int>>()),
                        "Missing unique_ptr deleter (as_unique_ptr).");
}

TEST_CASE("from_raw_ptr_unowned+as_shared_ptr", "[S]") {
    static int value = 19;
    auto hld = smart_holder::from_raw_ptr_unowned(&value);
    REQUIRE(*hld.as_shared_ptr<int>() == 19);
}

TEST_CASE("from_raw_ptr_take_ownership+as_lvalue_ref", "[S]") {
    auto hld = smart_holder::from_raw_ptr_take_ownership(new int(19));
    REQUIRE(hld.has_pointee());
    REQUIRE(hld.as_lvalue_ref<int>() == 19);
}

TEST_CASE("from_raw_ptr_take_ownership+as_raw_ptr_release_ownership1", "[S]") {
    auto hld = smart_holder::from_raw_ptr_take_ownership(new int(19));
    auto new_owner = std::unique_ptr<int>(hld.as_raw_ptr_release_ownership<int>());
    REQUIRE(!hld.has_pointee());
    REQUIRE(*new_owner == 19);
}

TEST_CASE("from_raw_ptr_take_ownership+as_raw_ptr_release_ownership2", "[E]") {
    auto hld = smart_holder::from_raw_ptr_take_ownership(new int(19));
    auto shd_ptr = hld.as_shared_ptr<int>();
    REQUIRE_THROWS_WITH(hld.as_raw_ptr_release_ownership<int>(),
                        "Cannot disown use_count != 1 (as_raw_ptr_release_ownership).");
}

TEST_CASE("from_raw_ptr_take_ownership+as_unique_ptr1", "[S]") {
    auto hld = smart_holder::from_raw_ptr_take_ownership(new int(19));
    std::unique_ptr<int> new_owner = hld.as_unique_ptr<int>();
    REQUIRE(!hld.has_pointee());
    REQUIRE(*new_owner == 19);
}

TEST_CASE("from_raw_ptr_take_ownership+as_unique_ptr2", "[E]") {
    auto hld = smart_holder::from_raw_ptr_take_ownership(new int(19));
    auto shd_ptr = hld.as_shared_ptr<int>();
    REQUIRE_THROWS_WITH(hld.as_unique_ptr<int>(), "Cannot disown use_count != 1 (as_unique_ptr).");
}

TEST_CASE("from_raw_ptr_take_ownership+as_unique_ptr_with_deleter", "[E]") {
    auto hld = smart_holder::from_raw_ptr_take_ownership(new int(19));
    REQUIRE_THROWS_WITH((hld.as_unique_ptr<int, helpers::functor_builtin_delete<int>>()),
                        "Missing unique_ptr deleter (as_unique_ptr).");
}

TEST_CASE("from_raw_ptr_take_ownership+as_shared_ptr", "[S]") {
    auto hld = smart_holder::from_raw_ptr_take_ownership(new int(19));
    std::shared_ptr<int> new_owner = hld.as_shared_ptr<int>();
    REQUIRE(hld.has_pointee());
    REQUIRE(*new_owner == 19);
}

TEST_CASE("from_raw_ptr_take_ownership+disown+reclaim_disowned", "[S]") {
    auto hld = smart_holder::from_raw_ptr_take_ownership(new int(19));
    std::unique_ptr<int> new_owner(hld.as_raw_ptr_unowned<int>());
    hld.disown();
    REQUIRE(hld.as_lvalue_ref<int>() == 19);
    REQUIRE(*new_owner == 19);
    hld.reclaim_disowned(); // Manually veriified: without this, clang++ -fsanitize=address reports
                            // "detected memory leaks".
    (void) new_owner.release(); // Manually verified: without this, clang++ -fsanitize=address
                                // reports "attempting double-free".
    REQUIRE(hld.as_lvalue_ref<int>() == 19);
    REQUIRE(new_owner.get() == nullptr);
}

TEST_CASE("from_raw_ptr_take_ownership+disown+release_disowned", "[S]") {
    auto hld = smart_holder::from_raw_ptr_take_ownership(new int(19));
    std::unique_ptr<int> new_owner(hld.as_raw_ptr_unowned<int>());
    hld.disown();
    REQUIRE(hld.as_lvalue_ref<int>() == 19);
    REQUIRE(*new_owner == 19);
    hld.release_disowned();
    REQUIRE(!hld.has_pointee());
}

TEST_CASE("from_raw_ptr_take_ownership+disown+ensure_is_not_disowned", "[E]") {
    const char *context = "test_case";
    auto hld = smart_holder::from_raw_ptr_take_ownership(new int(19));
    hld.ensure_is_not_disowned(context); // Does not throw.
    std::unique_ptr<int> new_owner(hld.as_raw_ptr_unowned<int>());
    hld.disown();
    REQUIRE_THROWS_WITH(hld.ensure_is_not_disowned(context),
                        "Holder was disowned already (test_case).");
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
    REQUIRE_THROWS_WITH((hld.as_unique_ptr<int, helpers::functor_builtin_delete<int>>()),
                        "Incompatible unique_ptr deleter (as_unique_ptr).");
}

TEST_CASE("from_unique_ptr+as_shared_ptr", "[S]") {
    std::unique_ptr<int> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    std::shared_ptr<int> new_owner = hld.as_shared_ptr<int>();
    REQUIRE(hld.has_pointee());
    REQUIRE(*new_owner == 19);
}

TEST_CASE("from_unique_ptr_derived+as_unique_ptr_base", "[S]") {
    std::unique_ptr<helpers::derived> orig_owner(new helpers::derived());
    auto hld = smart_holder::from_unique_ptr(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    std::unique_ptr<helpers::base> new_owner = hld.as_unique_ptr<helpers::base>();
    REQUIRE(!hld.has_pointee());
    REQUIRE(new_owner->get() == 100);
}

TEST_CASE("from_unique_ptr_derived+as_unique_ptr_base2", "[E]") {
    std::unique_ptr<helpers::derived, helpers::functor_other_delete<helpers::derived>> orig_owner(
        new helpers::derived());
    auto hld = smart_holder::from_unique_ptr(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    REQUIRE_THROWS_WITH(
        (hld.as_unique_ptr<helpers::base, helpers::functor_builtin_delete<helpers::base>>()),
        "Incompatible unique_ptr deleter (as_unique_ptr).");
}

TEST_CASE("from_unique_ptr_with_deleter+as_lvalue_ref", "[S]") {
    std::unique_ptr<int, helpers::functor_builtin_delete<int>> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    REQUIRE(hld.as_lvalue_ref<int>() == 19);
}

TEST_CASE("from_unique_ptr_with_std_function_deleter+as_lvalue_ref", "[S]") {
    std::unique_ptr<int, std::function<void(const int *)>> orig_owner(
        new int(19), [](const int *raw_ptr) { delete raw_ptr; });
    auto hld = smart_holder::from_unique_ptr(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    REQUIRE(hld.as_lvalue_ref<int>() == 19);
}

TEST_CASE("from_unique_ptr_with_deleter+as_raw_ptr_release_ownership", "[E]") {
    std::unique_ptr<int, helpers::functor_builtin_delete<int>> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    REQUIRE_THROWS_WITH(hld.as_raw_ptr_release_ownership<int>(),
                        "Cannot disown custom deleter (as_raw_ptr_release_ownership).");
}

TEST_CASE("from_unique_ptr_with_deleter+as_unique_ptr", "[E]") {
    std::unique_ptr<int, helpers::functor_builtin_delete<int>> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    REQUIRE_THROWS_WITH(hld.as_unique_ptr<int>(),
                        "Incompatible unique_ptr deleter (as_unique_ptr).");
}

TEST_CASE("from_unique_ptr_with_deleter+as_unique_ptr_with_deleter1", "[S]") {
    std::unique_ptr<int, helpers::functor_builtin_delete<int>> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    std::unique_ptr<int, helpers::functor_builtin_delete<int>> new_owner
        = hld.as_unique_ptr<int, helpers::functor_builtin_delete<int>>();
    REQUIRE(!hld.has_pointee());
    REQUIRE(*new_owner == 19);
}

TEST_CASE("from_unique_ptr_with_deleter+as_unique_ptr_with_deleter2", "[E]") {
    std::unique_ptr<int, helpers::functor_builtin_delete<int>> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr(std::move(orig_owner));
    REQUIRE(orig_owner.get() == nullptr);
    REQUIRE_THROWS_WITH((hld.as_unique_ptr<int, helpers::functor_other_delete<int>>()),
                        "Incompatible unique_ptr deleter (as_unique_ptr).");
}

TEST_CASE("from_unique_ptr_with_deleter+as_shared_ptr", "[S]") {
    std::unique_ptr<int, helpers::functor_builtin_delete<int>> orig_owner(new int(19));
    auto hld = smart_holder::from_unique_ptr(std::move(orig_owner));
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
    auto hld = smart_holder::from_shared_ptr(orig_owner);
    REQUIRE_THROWS_WITH((hld.as_unique_ptr<int, helpers::functor_builtin_delete<int>>()),
                        "Missing unique_ptr deleter (as_unique_ptr).");
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

TEST_CASE("indestructible_int-from_raw_ptr_unowned+as_raw_ptr_unowned", "[S]") {
    using zombie = helpers::indestructible_int;
    // Using placement new instead of plain new, to not trigger leak sanitizer errors.
    static std::aligned_storage<sizeof(zombie), alignof(zombie)>::type memory_block[1];
    auto *value = new (memory_block) zombie(19);
    auto hld = smart_holder::from_raw_ptr_unowned(value);
    REQUIRE(hld.as_raw_ptr_unowned<zombie>()->valu == 19);
}

TEST_CASE("indestructible_int-from_raw_ptr_take_ownership", "[E]") {
    helpers::indestructible_int *value = nullptr;
    REQUIRE_THROWS_WITH(smart_holder::from_raw_ptr_take_ownership(value),
                        "Pointee is not destructible (from_raw_ptr_take_ownership).");
}

TEST_CASE("from_raw_ptr_take_ownership+as_shared_ptr-outliving_smart_holder", "[S]") {
    // Exercises guarded_builtin_delete flag_ptr validity past destruction of smart_holder.
    std::shared_ptr<int> longer_living;
    {
        auto hld = smart_holder::from_raw_ptr_take_ownership(new int(19));
        longer_living = hld.as_shared_ptr<int>();
    }
    REQUIRE(*longer_living == 19);
}

TEST_CASE("from_unique_ptr_with_deleter+as_shared_ptr-outliving_smart_holder", "[S]") {
    // Exercises guarded_custom_deleter flag_ptr validity past destruction of smart_holder.
    std::shared_ptr<int> longer_living;
    {
        std::unique_ptr<int, helpers::functor_builtin_delete<int>> orig_owner(new int(19));
        auto hld = smart_holder::from_unique_ptr(std::move(orig_owner));
        longer_living = hld.as_shared_ptr<int>();
    }
    REQUIRE(*longer_living == 19);
}
