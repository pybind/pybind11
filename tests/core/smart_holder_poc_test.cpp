#include "pybind11/smart_holder_poc.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using pybindit::memory::smart_holder;

namespace helpers {

template <typename T>
struct functor_builtin_delete {
  void operator()(T* ptr) { delete ptr; }
};

}  // namespace helpers

TEST_CASE("from_raw_ptr_unowned+as_raw_ptr_unowned", "[S]") {
  static int value = 19;
  smart_holder hld;
  hld.from_raw_ptr_unowned(&value);
  REQUIRE(*hld.as_raw_ptr_unowned<int>() == 19);
}

TEST_CASE("from_raw_ptr_unowned+const_value_ref", "[S]") {
  static int value = 19;
  smart_holder hld;
  hld.from_raw_ptr_unowned(&value);
  REQUIRE(hld.const_value_ref<int>() == 19);
}

TEST_CASE("from_raw_ptr_unowned+as_raw_ptr_release_ownership", "[E]") {}

TEST_CASE("from_raw_ptr_unowned+as_unique_ptr", "[E]") {}

TEST_CASE("from_raw_ptr_unowned+as_unique_ptr_with_deleter", "[E]") {}

TEST_CASE("from_raw_ptr_unowned+as_shared_ptr", "[S]") {
  static int value = 19;
  smart_holder hld;
  hld.from_raw_ptr_unowned(&value);
  REQUIRE(*hld.as_shared_ptr<int>() == 19);
}

TEST_CASE("from_raw_ptr_take_ownership+const_value_ref", "[S]") {
  smart_holder hld;
  hld.from_raw_ptr_take_ownership(new int(19));
  REQUIRE(hld.has_pointee());
  REQUIRE(hld.const_value_ref<int>() == 19);
}

TEST_CASE("from_raw_ptr_take_ownership+as_raw_ptr_release_ownership1", "[S]") {
  smart_holder hld;
  hld.from_raw_ptr_take_ownership(new int(19));
  auto new_owner =
      std::unique_ptr<int>(hld.as_raw_ptr_release_ownership<int>());
  REQUIRE(!hld.has_pointee());
  REQUIRE(*new_owner == 19);
}

TEST_CASE("from_raw_ptr_take_ownership+as_raw_ptr_release_ownership2", "[E]") {}

TEST_CASE("from_raw_ptr_take_ownership+as_unique_ptr1", "[S]") {
  smart_holder hld;
  hld.from_raw_ptr_take_ownership(new int(19));
  std::unique_ptr<int> new_owner = hld.as_unique_ptr<int>();
  REQUIRE(!hld.has_pointee());
  REQUIRE(*new_owner == 19);
}

TEST_CASE("from_raw_ptr_take_ownership+as_unique_ptr2", "[E]") {}

TEST_CASE("from_raw_ptr_take_ownership+as_unique_ptr_with_deleter", "[E]") {}

TEST_CASE("from_raw_ptr_take_ownership+as_shared_ptr", "[S]") {
  smart_holder hld;
  hld.from_raw_ptr_take_ownership(new int(19));
  std::shared_ptr<int> new_owner = hld.as_shared_ptr<int>();
  REQUIRE(hld.has_pointee());
  REQUIRE(*new_owner == 19);
}

TEST_CASE("from_unique_ptr+const_value_ref", "[S]") {
  std::unique_ptr<int> orig_owner(new int(19));
  smart_holder hld;
  hld.from_unique_ptr(std::move(orig_owner));
  REQUIRE(orig_owner.get() == nullptr);
  REQUIRE(hld.const_value_ref<int>() == 19);
}

TEST_CASE("from_unique_ptr+as_raw_ptr_release_ownership1", "[S]") {
  std::unique_ptr<int> orig_owner(new int(19));
  smart_holder hld;
  hld.from_unique_ptr(std::move(orig_owner));
  REQUIRE(orig_owner.get() == nullptr);
  auto new_owner =
      std::unique_ptr<int>(hld.as_raw_ptr_release_ownership<int>());
  REQUIRE(!hld.has_pointee());
  REQUIRE(*new_owner == 19);
}

TEST_CASE("from_unique_ptr+as_raw_ptr_release_ownership2", "[E]") {}

TEST_CASE("from_unique_ptr+as_unique_ptr1", "[S]") {
  std::unique_ptr<int> orig_owner(new int(19));
  smart_holder hld;
  hld.from_unique_ptr(std::move(orig_owner));
  REQUIRE(orig_owner.get() == nullptr);
  std::unique_ptr<int> new_owner = hld.as_unique_ptr<int>();
  REQUIRE(!hld.has_pointee());
  REQUIRE(*new_owner == 19);
}

TEST_CASE("from_unique_ptr+as_unique_ptr2", "[E]") {}

TEST_CASE("from_unique_ptr+as_unique_ptr_with_deleter", "[E]") {}

TEST_CASE("from_unique_ptr+as_shared_ptr", "[S]") {
  std::unique_ptr<int> orig_owner(new int(19));
  smart_holder hld;
  hld.from_unique_ptr(std::move(orig_owner));
  REQUIRE(orig_owner.get() == nullptr);
  std::shared_ptr<int> new_owner = hld.as_shared_ptr<int>();
  REQUIRE(hld.has_pointee());
  REQUIRE(*new_owner == 19);
}

TEST_CASE("from_unique_ptr_with_deleter+const_value_ref", "[S]") {
  std::unique_ptr<int, helpers::functor_builtin_delete<int>> orig_owner(
      new int(19));
  smart_holder hld;
  hld.from_unique_ptr_with_deleter(std::move(orig_owner));
  REQUIRE(orig_owner.get() == nullptr);
  REQUIRE(hld.const_value_ref<int>() == 19);
}

TEST_CASE("from_unique_ptr_with_deleter+as_raw_ptr_release_ownership", "[E]") {}

TEST_CASE("from_unique_ptr_with_deleter+as_unique_ptr", "[E]") {}

TEST_CASE("from_unique_ptr_with_deleter+as_unique_ptr_with_deleter1", "[S]") {
  std::unique_ptr<int, helpers::functor_builtin_delete<int>> orig_owner(
      new int(19));
  smart_holder hld;
  hld.from_unique_ptr_with_deleter(std::move(orig_owner));
  REQUIRE(orig_owner.get() == nullptr);
  std::unique_ptr<int, helpers::functor_builtin_delete<int>> new_owner =
      hld.as_unique_ptr_with_deleter<int,
                                     helpers::functor_builtin_delete<int>>();
  REQUIRE(!hld.has_pointee());
  REQUIRE(*new_owner == 19);
}

TEST_CASE("from_unique_ptr_with_deleter+as_unique_ptr_with_deleter2", "[E]") {}

TEST_CASE("from_unique_ptr_with_deleter+as_shared_ptr", "[S]") {
  std::unique_ptr<int, helpers::functor_builtin_delete<int>> orig_owner(
      new int(19));
  smart_holder hld;
  hld.from_unique_ptr_with_deleter(std::move(orig_owner));
  REQUIRE(orig_owner.get() == nullptr);
  std::shared_ptr<int> new_owner = hld.as_shared_ptr<int>();
  REQUIRE(hld.has_pointee());
  REQUIRE(*new_owner == 19);
}

TEST_CASE("from_shared_ptr+const_value_ref", "[S]") {
  std::shared_ptr<int> orig_owner(new int(19));
  smart_holder hld;
  hld.from_shared_ptr(orig_owner);
  REQUIRE(hld.const_value_ref<int>() == 19);
}

TEST_CASE("from_shared_ptr+as_raw_ptr_release_ownership", "[E]") {}

TEST_CASE("from_shared_ptr+as_unique_ptr", "[E]") {}

TEST_CASE("from_shared_ptr+as_unique_ptr_with_deleter", "[E]") {}

TEST_CASE("from_shared_ptr+as_shared_ptr", "[S]") {
  std::shared_ptr<int> orig_owner(new int(19));
  smart_holder hld;
  hld.from_shared_ptr(orig_owner);
  REQUIRE(*hld.as_shared_ptr<int>() == 19);
}
