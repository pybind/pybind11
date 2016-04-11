#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

from example.issues import print_cchar, print_char
from example.issues import DispatchIssue, dispatch_issue_go

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
