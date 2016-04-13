from __future__ import print_function
import sys

sys.path.append('.')

from example import return_class_1
from example import return_class_2
from example import return_none

print(type(return_class_1()))
print(type(return_class_2()))
print(type(return_none()))
