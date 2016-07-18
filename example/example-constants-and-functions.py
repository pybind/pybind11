#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

from example import test_function
from example import some_constant
from example import EMyEnumeration
from example import EFirstEntry
from example import ExampleWithEnum
from example import return_bytes
from example import print_bytes

print(EMyEnumeration)
print(EMyEnumeration.EFirstEntry)
print(EMyEnumeration.ESecondEntry)
print(EFirstEntry)

print(test_function())
print(test_function(7))
print(test_function(EMyEnumeration.EFirstEntry))
print(test_function(EMyEnumeration.ESecondEntry))
print("enum->integer = %i" % int(EMyEnumeration.ESecondEntry))
print("integer->enum = %s" % str(EMyEnumeration(2)))

print("A constant = " + str(some_constant))

print(ExampleWithEnum.EMode)
print(ExampleWithEnum.EMode.EFirstMode)
print(ExampleWithEnum.EFirstMode)
ExampleWithEnum.test_function(ExampleWithEnum.EFirstMode)

print("Equality test 1: " + str(
    ExampleWithEnum.test_function(ExampleWithEnum.EFirstMode) ==
    ExampleWithEnum.test_function(ExampleWithEnum.EFirstMode)))

print("Inequality test 1: " + str(
    ExampleWithEnum.test_function(ExampleWithEnum.EFirstMode) !=
    ExampleWithEnum.test_function(ExampleWithEnum.EFirstMode)))

print("Equality test 2: " + str(
    ExampleWithEnum.test_function(ExampleWithEnum.EFirstMode) ==
    ExampleWithEnum.test_function(ExampleWithEnum.ESecondMode)))

print("Inequality test 2: " + str(
    ExampleWithEnum.test_function(ExampleWithEnum.EFirstMode) !=
    ExampleWithEnum.test_function(ExampleWithEnum.ESecondMode)))

x = {
        ExampleWithEnum.test_function(ExampleWithEnum.EFirstMode): 1,
        ExampleWithEnum.test_function(ExampleWithEnum.ESecondMode): 2
}

x[ExampleWithEnum.test_function(ExampleWithEnum.EFirstMode)] = 3
x[ExampleWithEnum.test_function(ExampleWithEnum.ESecondMode)] = 4
print("Hashing test = " + str(x))

print_bytes(return_bytes())
