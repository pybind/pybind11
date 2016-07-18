#!/usr/bin/env python
from __future__ import print_function
import sys, pydoc
sys.path.append('.')

import example
from example import ExamplePythonTypes

ExamplePythonTypes.value = 15
print(ExamplePythonTypes.value)
print(ExamplePythonTypes.value2)

try:
    ExamplePythonTypes()
except Exception as e:
    print(e)

try:
    ExamplePythonTypes.value2 = 15
except Exception as e:
    print(e)

instance = ExamplePythonTypes.new_instance()

dict_result = instance.get_dict()
dict_result['key2'] = 'value2'
instance.print_dict(dict_result)

dict_result = instance.get_dict_2()
dict_result['key2'] = 'value2'
instance.print_dict_2(dict_result)

set_result = instance.get_set()
set_result.add('key3')
instance.print_set(set_result)

set_result = instance.get_set2()
set_result.add('key3')
instance.print_set_2(set_result)

list_result = instance.get_list()
list_result.append('value2')
instance.print_list(list_result)

list_result = instance.get_list_2()
list_result.append('value2')
instance.print_list_2(list_result)

array_result = instance.get_array()
print(array_result)
instance.print_array(array_result)

try:
    instance.throw_exception()
except Exception as e:
    print(e)

print(instance.pair_passthrough((True, "test")))
print(instance.tuple_passthrough((True, "test", 5)))

print(pydoc.render_doc(ExamplePythonTypes, "Help on %s"))

print("__name__(example) = %s" % example.__name__)
print("__name__(example.ExamplePythonTypes) = %s" % ExamplePythonTypes.__name__)
print("__module__(example.ExamplePythonTypes) = %s" % ExamplePythonTypes.__module__)
print("__name__(example.ExamplePythonTypes.get_set) = %s" % ExamplePythonTypes.get_set.__name__)
print("__module__(example.ExamplePythonTypes.get_set) = %s" % ExamplePythonTypes.get_set.__module__)
