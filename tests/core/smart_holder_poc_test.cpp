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

TEST_CASE("from_raw_ptr_take_ownership+const_value_ref", "[feasible]") {
  smart_holder hld;
  REQUIRE(!hld.has_pointee());
  hld.from_raw_ptr_take_ownership(new int(19));
  REQUIRE(hld.has_pointee());
  REQUIRE(hld.const_value_ref<int>() == 19);
}

TEST_CASE("from_raw_ptr_unowned+const_value_ref", "[feasible]") {
  static int value = 19;
  smart_holder hld;
  hld.from_raw_ptr_unowned(&value);
  REQUIRE(hld.const_value_ref<int>() == 19);
}

TEST_CASE("from_raw_ptr_take_ownership+as_raw_ptr_release_ownership", "[feasible]") {
  smart_holder hld;
  hld.from_raw_ptr_take_ownership(new int(19));
  auto new_owner =
      std::unique_ptr<int>(hld.as_raw_ptr_release_ownership<int>());
  REQUIRE(!hld.has_pointee());
}

TEST_CASE("from_raw_ptr_take_ownership+as_raw_ptr_unowned", "[feasible]") {
  smart_holder hld;
  hld.from_raw_ptr_take_ownership(new int(19));
  int* raw_ptr = hld.as_raw_ptr_unowned<int>();
  REQUIRE(hld.has_pointee());
  REQUIRE(*raw_ptr == 19);
}

TEST_CASE("from_unique_ptr+const_value_ref+const_value_ref", "[feasible]") {
  std::unique_ptr<int> orig_owner(new int(19));
  smart_holder hld;
  hld.from_unique_ptr(std::move(orig_owner));
  REQUIRE(orig_owner.get() == nullptr);
  REQUIRE(hld.const_value_ref<int>() == 19);
}

TEST_CASE("from_raw_ptr_take_ownership+as_unique_ptr", "[feasible]") {
  smart_holder hld;
  hld.from_raw_ptr_take_ownership(new int(19));
  auto new_owner = hld.as_unique_ptr<int>();
  REQUIRE(!hld.has_pointee());
  REQUIRE(*new_owner == 19);
}

TEST_CASE("from_unique_ptr_with_deleter+const_value_ref", "[feasible]") {
  std::unique_ptr<int, helpers::functor_builtin_delete<int>> orig_owner(
      new int(19));
  smart_holder hld;
  hld.from_unique_ptr_with_deleter(std::move(orig_owner));
  REQUIRE(orig_owner.get() == nullptr);
  REQUIRE(hld.const_value_ref<int>() == 19);
}

TEST_CASE("from_unique_ptr_with_deleter+as_unique_ptr_with_deleter", "[feasible]") {
  std::unique_ptr<int, helpers::functor_builtin_delete<int>> orig_owner(
      new int(19));
  smart_holder hld;
  hld.from_unique_ptr_with_deleter(std::move(orig_owner));
  auto new_owner =
      hld.as_unique_ptr_with_deleter<int,
                                     helpers::functor_builtin_delete<int>>();
  REQUIRE(!hld.has_pointee());
  REQUIRE(*new_owner == 19);
}

TEST_CASE("from_shared_ptr+const_value_ref", "[feasible]") {
  std::shared_ptr<int> orig_owner(new int(19));
  smart_holder hld;
  hld.from_shared_ptr(orig_owner);
  REQUIRE(orig_owner.get() != nullptr);
  REQUIRE(hld.const_value_ref<int>() == 19);
}

TEST_CASE("from_raw_ptr_take_ownership+as_shared_ptr", "[feasible]") {
  smart_holder hld;
  hld.from_raw_ptr_take_ownership(new int(19));
  auto new_owner = hld.as_shared_ptr<int>();
  REQUIRE(hld.has_pointee());
  REQUIRE(*new_owner == 19);
}
