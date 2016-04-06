from __future__ import print_function
import sys

sys.path.append('.')

from example import StringList, print_opaque_list
from example import return_void_ptr, print_void_ptr
from example import return_null_str, print_null_str
from example import return_unique_ptr

l = StringList()
l.push_back("Element 1")
l.push_back("Element 2")
print_opaque_list(l)
print("Back element is %s" % l.back())
l.pop_back()
print_opaque_list(l)

print_void_ptr(return_void_ptr())

print(return_null_str())
print_null_str(return_null_str())

print(return_unique_ptr())
