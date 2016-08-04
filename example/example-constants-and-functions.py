#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

from example import test_function
from example import some_constant
from example import EMyEnumeration
from example import ECMyEnum, test_ecenum
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
test_ecenum(ECMyEnum.Three)
z = ECMyEnum.Two
test_ecenum(z)
try:
    z == 2
    print("Bad: expected a TypeError exception")
except TypeError:
    try:
        z != 3
        print("Bad: expected a TypeError exception")
    except TypeError:
        print("Good: caught expected TypeError exceptions for scoped enum ==/!= int comparisons")

y = EMyEnumeration.ESecondEntry
try:
    y == 2
    y != 2
    print("Good: no TypeError exception for unscoped enum ==/!= int comparisions")
except TypeError:
    print("Bad: caught TypeError exception for unscoped enum ==/!= int comparisons")

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

print("Equality test 3: " + str(
    ExampleWithEnum.test_function(ExampleWithEnum.EFirstMode) ==
    int(ExampleWithEnum.test_function(ExampleWithEnum.EFirstMode))))

print("Inequality test 3: " + str(
    ExampleWithEnum.test_function(ExampleWithEnum.EFirstMode) !=
    int(ExampleWithEnum.test_function(ExampleWithEnum.EFirstMode))))

print("Equality test 4: " + str(
    ExampleWithEnum.test_function(ExampleWithEnum.EFirstMode) ==
    int(ExampleWithEnum.test_function(ExampleWithEnum.ESecondMode))))

print("Inequality test 4: " + str(
    ExampleWithEnum.test_function(ExampleWithEnum.EFirstMode) !=
    int(ExampleWithEnum.test_function(ExampleWithEnum.ESecondMode))))

x = {
        ExampleWithEnum.test_function(ExampleWithEnum.EFirstMode): 1,
        ExampleWithEnum.test_function(ExampleWithEnum.ESecondMode): 2
}

x[ExampleWithEnum.test_function(ExampleWithEnum.EFirstMode)] = 3
x[ExampleWithEnum.test_function(ExampleWithEnum.ESecondMode)] = 4
print("Hashing test = " + str(x))

print_bytes(return_bytes())
