from __future__ import print_function
import sys
import gc
sys.path.append('.')

from example import Parent, Child

if True:
    p = Parent()
    p.addChild(Child())
    gc.collect()
    print(p)
    p = None

gc.collect()
print("")

if True:
    p = Parent()
    p.returnChild()
    gc.collect()
    print(p)
    p = None

gc.collect()
print("")

if True:
    p = Parent()
    p.addChildKeepAlive(Child())
    gc.collect()
    print(p)
    p = None
gc.collect()
print("")

if True:
    p = Parent()
    p.returnChildKeepAlive()
    gc.collect()
    print(p)
    p = None

gc.collect()
print("")
print("Terminating..")
