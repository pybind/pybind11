#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

from example import Example12, runExample12, runExample12Virtual, runExample12Bool


class ExtendedExample12(Example12):
    def __init__(self, state):
        super(ExtendedExample12, self).__init__(state + 1)
        self.data = "Hello world"

    def run(self, value):
        print('ExtendedExample12::run(%i), calling parent..' % value)
        return super(ExtendedExample12, self).run(value + 1)

    def run_bool(self):
        print('ExtendedExample12::run_bool()')
        return False

    def pure_virtual(self):
        print('ExtendedExample12::pure_virtual(): %s' % self.data)


ex12 = Example12(10)
print(runExample12(ex12, 20))
try:
    runExample12Virtual(ex12)
except Exception as e:
    print("Caught expected exception: " + str(e))

ex12p = ExtendedExample12(10)
print(runExample12(ex12p, 20))
print(runExample12Bool(ex12p))
runExample12Virtual(ex12p)
