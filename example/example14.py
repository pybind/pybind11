from __future__ import print_function
import sys

sys.path.append('.')

from example import StringList, print_opaque_list
from example import ClassWithSTLVecProperty
from example import return_void_ptr, print_void_ptr
from example import return_null_str, print_null_str
from example import return_unique_ptr
from example import Example1

#####

l = StringList()
l.push_back("Element 1")
l.push_back("Element 2")
print_opaque_list(l)
print("Back element is %s" % l.back())
for i, k in enumerate(l):
    print("%i/%i : %s" % (i + 1, len(l), k))
l.pop_back()
print_opaque_list(l)

#####
cvp = ClassWithSTLVecProperty()
print_opaque_list(cvp.stringList)

cvp.stringList = l
cvp.stringList.push_back("Element 3")
print_opaque_list(cvp.stringList)

#####

print_void_ptr(return_void_ptr())
print_void_ptr(Example1())  # Should also work for other C++ types

try:
    print_void_ptr([1, 2, 3])  # This should not work
except Exception as e:
    print("Caught expected exception: " + str(e))

print(return_null_str())
print_null_str(return_null_str())

#####

ptr = return_unique_ptr()
print(ptr)
print_opaque_list(ptr)
