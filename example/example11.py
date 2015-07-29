#!/usr/bin/env python3
import sys, pydoc
sys.path.append('.')

import example

from example import kw_func
from example import kw_func2

print(pydoc.render_doc(kw_func, "Help on %s"))
print(pydoc.render_doc(kw_func2, "Help on %s"))

kw_func(5, 10)
kw_func(5, y = 10)
kw_func(y = 10, x = 5)

kw_func2()

kw_func2(5)
kw_func2(x=5)

kw_func2(y=10)

kw_func2(5, 10)
kw_func2(x=5, y=10)
