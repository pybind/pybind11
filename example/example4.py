#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

from example import test_function
from example import some_constant
from example import EMyEnumeration
from example import EFirstEntry
from example import Example4
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

print(Example4.EMode)
print(Example4.EMode.EFirstMode)
print(Example4.EFirstMode)
Example4.test_function(Example4.EFirstMode)

print("Equality test 1: " + str(
    Example4.test_function(Example4.EFirstMode) ==
    Example4.test_function(Example4.EFirstMode)))

print("Inequality test 1: " + str(
    Example4.test_function(Example4.EFirstMode) !=
    Example4.test_function(Example4.EFirstMode)))

print("Equality test 2: " + str(
    Example4.test_function(Example4.EFirstMode) ==
    Example4.test_function(Example4.ESecondMode)))

print("Inequality test 2: " + str(
    Example4.test_function(Example4.EFirstMode) !=
    Example4.test_function(Example4.ESecondMode)))

x = {
        Example4.test_function(Example4.EFirstMode): 1,
        Example4.test_function(Example4.ESecondMode): 2
}

x[Example4.test_function(Example4.EFirstMode)] = 3
x[Example4.test_function(Example4.ESecondMode)] = 4
print("Hashing test = " + str(x))

print_bytes(return_bytes())
