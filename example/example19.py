#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

import example

print("Can we catch a MyException?")
try:
    example.throws1()
except example.MyException as e:
    print(e.__class__.__name__, ":", e)
print("")

print("Can we translate to standard Python exceptions?")
try:
    example.throws2()
except Exception as e:
    print(e.__class__.__name__, ":", e)
print("")

print("Can we handle unknown exceptions?")
try:
    example.throws3()
except Exception as e:
    print(e.__class__.__name__, ":", e)
print("")

print("Can we delegate to another handler by rethrowing?")
try:
    example.throws4()
except example.MyException as e:
    print(e.__class__.__name__, ":", e)
print("")

print("Can we fall-through to the default handler?")
try:
    example.throws_logic_error()
except Exception as e:
    print(e.__class__.__name__, ":", e)
print("")

