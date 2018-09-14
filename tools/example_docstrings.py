"""Example documentation strings - module documentation.

Created on 26 Feb 2018

@author: paulross
"""
#: Global variable with Sphinx style documentation.
#: We don't get to see this.
global_variable = 4

def global_function(a, b, c):
    """global_function documentation."""

class OuterClass(object):
    """OuterClass documentation"""

    def __init__(self, *args, **kwargs):
        """OuterClass constructor documentation"""
        pass
        
    def method(self, *args, **kwargs):
        """OuterClass method documentation"""
        def function_inside_method():
            """OuterClass.method.function_inside_method() documentation.
            We don't get to see this."""
            pass
        pass
        
    class InnerClass(object):
        """InnerClass documentation"""
    
        def __init__(self, *args, **kwargs):
            """InnerClass constructor documentation"""
            pass
            
        def method(self, *args, **kwargs):
            """InnerClass method documentation"""
            pass
