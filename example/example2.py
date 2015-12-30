#!/usr/bin/env python
from __future__ import print_function
import six
import sys, pydoc
sys.path.append('.')

from example import Example2

Example2.value = 15
print(Example2.value)
print(Example2.value2)

try:
    Example2()
except Exception as e:
    print(e)

try:
    Example2.value2 = 15
except Exception as e:
    print(e)

instance = Example2.new_instance()

dict_result = instance.get_dict()
dict_result['key2'] = 'value2'
instance.print_dict(dict_result)

dict_result = instance.get_dict_2()
dict_result['key2'] = 'value2'
instance.print_dict_2(dict_result)

set_result = instance.get_set()
set_result.add(six.u('key3'))
instance.print_set(set_result)

set_result = instance.get_set2()
set_result.add(six.u('key3'))
instance.print_set_2(set_result)

list_result = instance.get_list()
list_result.append('value2')
instance.print_list(list_result)

list_result = instance.get_list_2()
list_result.append('value2')
instance.print_list_2(list_result)

try:
    instance.throw_exception()
except Exception as e:
    print(e)

print(instance.pair_passthrough((True, "test")))
print(instance.tuple_passthrough((True, "test", 5)))

print(pydoc.render_doc(Example2, "Help on %s"))
