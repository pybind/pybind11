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

test_cleanup()
