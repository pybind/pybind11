#pragma once

#include <memory>
#include <test_cross_module_rtti_lib_export.h>

#if defined(_MSC_VER)
__pragma(warning(disable : 4251))
#endif

    namespace lib {

    class TEST_CROSS_MODULE_RTTI_LIB_EXPORT Base : public std::enable_shared_from_this<Base> {
    public:
        Base(int a, int b);
        virtual ~Base() = default;

        virtual int get() const;

        int a;
        int b;
    };

    class TEST_CROSS_MODULE_RTTI_LIB_EXPORT Foo : public Base {
    public:
        Foo(int a, int b);

        int get() const override;
    };

} // namespace lib
