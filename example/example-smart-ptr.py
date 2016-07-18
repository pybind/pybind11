#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

from example import MyObject1
from example import MyObject2
from example import MyObject3

from example import make_object_1
from example import make_object_2
from example import make_myobject1_1
from example import make_myobject1_2
from example import make_myobject2_1
from example import make_myobject2_2
from example import make_myobject3_1
from example import make_myobject3_2

from example import print_object_1
from example import print_object_2
from example import print_object_3
from example import print_object_4

from example import print_myobject1_1
from example import print_myobject1_2
from example import print_myobject1_3
from example import print_myobject1_4

from example import print_myobject2_1
from example import print_myobject2_2
from example import print_myobject2_3
from example import print_myobject2_4

from example import print_myobject3_1
from example import print_myobject3_2
from example import print_myobject3_3
from example import print_myobject3_4

for o in [make_object_1(), make_object_2(), MyObject1(3)]:
    print("Reference count = %i" % o.getRefCount())
    print_object_1(o)
    print_object_2(o)
    print_object_3(o)
    print_object_4(o)

for o in [make_myobject1_1(), make_myobject1_2(), MyObject1(6), 7]:
    print(o)
    if not isinstance(o, int):
        print_object_1(o)
        print_object_2(o)
        print_object_3(o)
        print_object_4(o)
    print_myobject1_1(o)
    print_myobject1_2(o)
    print_myobject1_3(o)
    print_myobject1_4(o)

for o in [MyObject2(8), make_myobject2_1(), make_myobject2_2()]:
    print(o)
    print_myobject2_1(o)
    print_myobject2_2(o)
    print_myobject2_3(o)
    print_myobject2_4(o)

for o in [MyObject3(9), make_myobject3_1(), make_myobject3_2()]:
    print(o)
    print_myobject3_1(o)
    print_myobject3_2(o)
    print_myobject3_3(o)
    print_myobject3_4(o)
