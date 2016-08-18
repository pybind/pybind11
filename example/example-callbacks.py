#!/usr/bin/env python
from __future__ import print_function
from functools import partial
import sys
sys.path.append('.')

from example import Pet
from example import Dog
from example import Rabbit
from example import dog_bark
from example import pet_print

polly = Pet('Polly', 'parrot')
molly = Dog('Molly')
roger = Rabbit('Rabbit')
print(roger.name() + " is a " + roger.species())
pet_print(roger)
print(polly.name() + " is a " + polly.species())
pet_print(polly)
print(molly.name() + " is a " + molly.species())
pet_print(molly)
dog_bark(molly)
try:
    dog_bark(polly)
except Exception as e:
    print('The following error is expected: ' + str(e))

from example import test_callback1
from example import test_callback2
from example import test_callback3
from example import test_callback4
from example import test_callback5
from example import test_callback6
from example import test_cleanup

def func1():
    print('Callback function 1 called!')

def func2(a, b, c, d):
    print('Callback function 2 called : ' + str(a) + ", " + str(b) + ", " + str(c) + ", "+ str(d))
    return d

def func3(a):
    print('Callback function 3 called : ' + str(a))

print(test_callback1(func1))
print(test_callback2(func2))
print(test_callback1(partial(func2, "Hello", "from", "partial", "object")))
print(test_callback1(partial(func3, "Partial object with one argument")))

test_callback3(lambda i: i + 1)
f = test_callback4()
print("func(43) = %i" % f(43))
f = test_callback5()
print("func(number=43) = %i" % f(number=43))
test_callback6(lambda i: i + 2)
test_callback6()
test_callback6(None)

test_cleanup()

from example import payload_cstats
cstats = payload_cstats()
print("Payload instances not destroyed:", cstats.alive())
print("Copy constructions:", cstats.copy_constructions)
print("Move constructions:", cstats.move_constructions >= 1)

from example import dummy_function
from example import dummy_function2
from example import test_dummy_function
from example import roundtrip

test_dummy_function(dummy_function)
test_dummy_function(roundtrip(dummy_function))
test_dummy_function(lambda x: x + 2)

try:
    test_dummy_function(dummy_function2)
    print("Problem!")
except Exception as e:
    if 'Incompatible function arguments' in str(e):
        print("All OK!")
    else:
        print("Problem!")

try:
    test_dummy_function(lambda x, y: x + y)
    print("Problem!")
except Exception as e:
    if 'missing 1 required positional argument' in str(e) or \
       'takes exactly 2 arguments' in str(e):
        print("All OK!")
    else:
        print("Problem!")

print(test_callback3.__doc__)
print(test_callback4.__doc__)
