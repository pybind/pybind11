#include <lib.h>

namespace lib {

Base::Base(int a, int b) : a(a), b(b) {}

int Base::get() const { return a + b; }

Foo::Foo(int a, int b) : Base{a, b} {}

int Foo::get() const { return 2 * a + b; }

} // namespace lib
