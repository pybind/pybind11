#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

from example.issues import print_cchar, print_char
from example.issues import DispatchIssue, dispatch_issue_go
from example.issues import Placeholder, return_vec_of_reference_wrapper
from example.issues import iterator_passthrough
from example.issues import ElementList, ElementA, print_element
from example.issues import expect_float, expect_int
from example.issues import A, call_f
import gc

print_cchar("const char *")
print_char('c')


class PyClass1(DispatchIssue):
    def dispatch(self):
        print("Yay..")


class PyClass2(DispatchIssue):
    def dispatch(self):
        try:
            super(PyClass2, self).dispatch()
        except Exception as e:
            print("Failed as expected: " + str(e))
        p = PyClass1()
        dispatch_issue_go(p)

b = PyClass2()
dispatch_issue_go(b)

print(return_vec_of_reference_wrapper(Placeholder(4)))

print(list(iterator_passthrough(iter([3, 5, 7, 9, 11, 13, 15]))))

el = ElementList()
for i in range(10):
    el.add(ElementA(i))
gc.collect()
for i, v in enumerate(el.get()):
    print("%i==%i, " % (i, v.value()), end='')
print()

try:
    print_element(None)
except Exception as e:
    print("Failed as expected: " + str(e))

try:
    print(expect_int(5.2))
except Exception as e:
    print("Failed as expected: " + str(e))

print(expect_float(12))

class B(A):
    def __init__(self):
        super(B, self).__init__()

    def f(self):
        print("In python f()")

print("C++ version")
a = A()
call_f(a)

print("Python version")
b = B()
call_f(b)

