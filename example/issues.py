#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

from example.issues import print_cchar, print_char
from example.issues import DispatchIssue, dispatch_issue_go
from example.issues import Placeholder, return_vec_of_reference_wrapper
from example.issues import iterator_passthrough
from example.issues import ElementList, ElementA
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
