#!/usr/bin/env python
from __future__ import print_function
import sys
sys.path.append('.')

from example import Sequence

s = Sequence(5)
print("s = " + str(s))
print("len(s) = " + str(len(s)))
print("s[0], s[3] = %f %f" % (s[0], s[3]))
print('12.34 in s: ' + str(12.34 in s))
s[0], s[3] = 12.34, 56.78
print('12.34 in s: ' + str(12.34 in s))
print("s[0], s[3] = %f %f" % (s[0], s[3]))
rev = reversed(s)
rev2 = s[::-1]
print("rev[0], rev[1], rev[2], rev[3], rev[4] = %f %f %f %f %f" % (rev[0], rev[1], rev[2], rev[3], rev[4]))

for i in rev:
    print(i, end=' ')
print('')
for i in rev2:
    print(i, end=' ')
print('')
print(rev == rev2)
rev[0::2] = Sequence([2.0, 2.0, 2.0])
for i in rev:
    print(i, end=' ')
print('')
