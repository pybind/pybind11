#!/usr/bin/env python
from __future__ import print_function
import sys
import pydoc

sys.path.append('.')

from example import kw_func0, kw_func1, kw_func2, kw_func3, kw_func4, call_kw_func
from example import args_function, args_kwargs_function, kw_func_udl, kw_func_udl_z
from example import KWClass

print(pydoc.render_doc(kw_func0, "Help on %s"))
print(pydoc.render_doc(kw_func1, "Help on %s"))
print(pydoc.render_doc(kw_func2, "Help on %s"))
print(pydoc.render_doc(kw_func3, "Help on %s"))
print(pydoc.render_doc(kw_func4, "Help on %s"))
print(pydoc.render_doc(kw_func_udl, "Help on %s"))
print(pydoc.render_doc(kw_func_udl_z, "Help on %s"))
print(pydoc.render_doc(args_function, "Help on %s"))
print(pydoc.render_doc(args_kwargs_function, "Help on %s"))

print(KWClass.foo0.__doc__)
print(KWClass.foo1.__doc__)

kw_func1(5, 10)
kw_func1(5, y=10)
kw_func1(y=10, x=5)

kw_func2()

kw_func2(5)
kw_func2(x=5)

kw_func2(y=10)

kw_func2(5, 10)
kw_func2(x=5, y=10)

try:
    kw_func2(x=5, y=10, z=12)
except Exception as e:
    print("Caught expected exception: " + str(e))

kw_func4()
kw_func4(myList=[1, 2, 3])

call_kw_func(kw_func2)

args_function('arg1_value', 'arg2_value', 3)
args_kwargs_function('arg1_value', 'arg2_value', arg3='arg3_value', arg4=4)

kw_func_udl(x=5, y=10)
kw_func_udl_z(x=5)
