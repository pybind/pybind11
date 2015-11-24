#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

from example import MyObject
from example import make_object_1
from example import make_object_2
from example import make_myobject_4
from example import make_myobject_5
from example import make_myobject2_1
from example import make_myobject2_2
from example import print_object_1
from example import print_object_2
from example import print_object_3
from example import print_object_4
from example import print_myobject_1
from example import print_myobject_2
from example import print_myobject_3
from example import print_myobject_4
from example import print_myobject2_1
from example import print_myobject2_2
from example import print_myobject2_3
from example import print_myobject2_4

for o in [make_object_1(), make_object_2(), MyObject(3)]:
    print("Reference count = %i" % o.getRefCount())
    print_object_1(o)
    print_object_2(o)
    print_object_3(o)
    print_object_4(o)

for o in [make_myobject_4(), make_myobject_5(), MyObject(6), 7]:
    print(o)
    if not isinstance(o, int):
        print_object_1(o)
        print_object_2(o)
        print_object_3(o)
        print_object_4(o)
    print_myobject_1(o)
    print_myobject_2(o)
    print_myobject_3(o)
    print_myobject_4(o)


for o in [make_myobject2_1(), make_myobject2_2()]:
    print(o)
    print_myobject2_1(o)
    print_myobject2_2(o)
    print_myobject2_3(o)
    print_myobject2_4(o)
