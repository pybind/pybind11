#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

from example import ExampleVirt, runExampleVirt, runExampleVirtVirtual, runExampleVirtBool
from example import A_Repeat, B_Repeat, C_Repeat, D_Repeat, A_Tpl, B_Tpl, C_Tpl, D_Tpl
from example import NCVirt, NonCopyable, Movable


class ExtendedExampleVirt(ExampleVirt):
    def __init__(self, state):
        super(ExtendedExampleVirt, self).__init__(state + 1)
        self.data = "Hello world"

    def run(self, value):
        print('ExtendedExampleVirt::run(%i), calling parent..' % value)
        return super(ExtendedExampleVirt, self).run(value + 1)

    def run_bool(self):
        print('ExtendedExampleVirt::run_bool()')
        return False

    def pure_virtual(self):
        print('ExtendedExampleVirt::pure_virtual(): %s' % self.data)


ex12 = ExampleVirt(10)
print(runExampleVirt(ex12, 20))
try:
    runExampleVirtVirtual(ex12)
except Exception as e:
    print("Caught expected exception: " + str(e))

ex12p = ExtendedExampleVirt(10)
print(runExampleVirt(ex12p, 20))
print(runExampleVirtBool(ex12p))
runExampleVirtVirtual(ex12p)

sys.stdout.flush()

class VI_AR(A_Repeat):
    def unlucky_number(self):
        return 99
class VI_AT(A_Tpl):
    def unlucky_number(self):
        return 999

class VI_CR(C_Repeat):
    def lucky_number(self):
        return C_Repeat.lucky_number(self) + 1.25
class VI_CT(C_Tpl):
    pass
class VI_CCR(VI_CR):
    def lucky_number(self):
        return VI_CR.lucky_number(self) * 10
class VI_CCT(VI_CT):
    def lucky_number(self):
        return VI_CT.lucky_number(self) * 1000


class VI_DR(D_Repeat):
    def unlucky_number(self):
        return 123
    def lucky_number(self):
        return 42.0
class VI_DT(D_Tpl):
    def say_something(self, times):
        print("VI_DT says:" + (' quack' * times))
    def unlucky_number(self):
        return 1234
    def lucky_number(self):
        return -4.25

classes = [
    # A_Repeat, A_Tpl, # abstract (they have a pure virtual unlucky_number)
    VI_AR, VI_AT,
    B_Repeat, B_Tpl,
    C_Repeat, C_Tpl,
    VI_CR, VI_CT, VI_CCR, VI_CCT,
    D_Repeat, D_Tpl, VI_DR, VI_DT
]

for cl in classes:
    print("\n%s:" % cl.__name__)
    obj = cl()
    obj.say_something(3)
    print("Unlucky = %d" % obj.unlucky_number())
    if hasattr(obj, "lucky_number"):
        print("Lucky = %.2f" % obj.lucky_number())

class NCVirtExt(NCVirt):
    def get_noncopyable(self, a, b):
        # Constructs and returns a new instance:
        nc = NonCopyable(a*a, b*b)
        return nc
    def get_movable(self, a, b):
        # Return a referenced copy
        self.movable = Movable(a, b)
        return self.movable

class NCVirtExt2(NCVirt):
    def get_noncopyable(self, a, b):
        # Keep a reference: this is going to throw an exception
        self.nc = NonCopyable(a, b)
        return self.nc
    def get_movable(self, a, b):
        # Return a new instance without storing it
        return Movable(a, b)

ncv1 = NCVirtExt()
print("2^2 * 3^2 =")
ncv1.print_nc(2, 3)
print("4 + 5 =")
ncv1.print_movable(4, 5)
ncv2 = NCVirtExt2()
print("7 + 7 =")
ncv2.print_movable(7, 7)
try:
    ncv2.print_nc(9, 9)
    print("Something's wrong: exception not raised!")
except RuntimeError as e:
    # Don't print the exception message here because it differs under debug/non-debug mode
    print("Caught expected exception")
