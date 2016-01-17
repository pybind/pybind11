#!/usr/bin/env python
from __future__ import print_function
import sys
import pydoc

sys.path.append('.')

from example import kw_func, kw_func2, kw_func3, kw_func4

print(pydoc.render_doc(kw_func, "Help on %s"))
print(pydoc.render_doc(kw_func2, "Help on %s"))
print(pydoc.render_doc(kw_func3, "Help on %s"))
print(pydoc.render_doc(kw_func4, "Help on %s"))

kw_func(5, 10)
kw_func(5, y=10)
kw_func(y=10, x=5)

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
kw_func4(myList = [1, 2, 3])
