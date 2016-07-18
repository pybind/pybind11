#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

from example import ExampleVirt, runExampleVirt, runExampleVirtVirtual, runExampleVirtBool


class ExtendedExampleVirt(ExampleVirt):
    def __init__(self, state):
        super(ExtendedExampleVirt, self).__init__(state + 1)
        self.data = "Hello world"

    def run(self, value):
        print('ExtendedExampleVirt::run(%i), calling parent..' % value)
        return super(ExtendedExampleVirt, self).run(value + 1)

    def run_bool(self):
        print('ExtendedExampleVirt::run_bool()')
        return False

    def pure_virtual(self):
        print('ExtendedExampleVirt::pure_virtual(): %s' % self.data)


ex12 = ExampleVirt(10)
print(runExampleVirt(ex12, 20))
try:
    runExampleVirtVirtual(ex12)
except Exception as e:
    print("Caught expected exception: " + str(e))

ex12p = ExtendedExampleVirt(10)
print(runExampleVirt(ex12p, 20))
print(runExampleVirtBool(ex12p))
runExampleVirtVirtual(ex12p)
