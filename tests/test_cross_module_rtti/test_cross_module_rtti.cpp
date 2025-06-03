
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <catch.hpp>
#include <lib.h>

static constexpr auto script = R"(
import test_cross_module_rtti_bindings

class Bar(test_cross_module_rtti_bindings.Base):
    def __init__(self, a, b):
        test_cross_module_rtti_bindings.Base.__init__(self, a, b)

    def get(self):
        return 4 * self.a + self.b


def get_bar(a, b):
    return Bar(a, b)

)";

TEST_CASE("Simple case where without is_alias") {
    // "Simple" case this will not have `python_instance_is_alias` set in type_cast_base.h:771
    auto bindings = pybind11::module_::import("test_cross_module_rtti_bindings");
    auto holder = bindings.attr("get_foo")(1, 2);
    auto foo = holder.cast<std::shared_ptr<lib::Base>>();
    REQUIRE(foo->get() == 4); // 2 * 1 + 2 = 4
}

TEST_CASE("Complex case where with it_alias") {
    // "Complex" case this will have `python_instance_is_alias` set in type_cast_base.h:771
    pybind11::exec(script);
    auto main = pybind11::module::import("__main__");

    // The critical part of "Bar" is that it will have the `is_alias` `instance` flag set.
    // I'm not quite sure what is required to get that flag, this code is derived from a
    // larger code where this issue was observed.
    auto holder2 = main.attr("get_bar")(1, 2);

    // this will trigger `std::get_deleter<memory::guarded_delete>` in type_cast_base.h:772
    // This will fail since the program will see two different typeids for `memory::guarded_delete`
    // on from the bindings module and one from "main", which will both have
    // `__is_type_name_unique` as true and but still have different values. Hence we will not find
    // the deleter and the cast fill fail. See "__eq(__type_name_t __lhs, __type_name_t __rhs)" in
    // typeinfo in libc++
    auto bar = holder2.cast<std::shared_ptr<lib::Base>>();
    REQUIRE(bar->get() == 6); // 4 * 1 + 2 = 6
}
