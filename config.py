#!/usr/bin/env python
# encoding: utf-8

import traceback
import sys

try:
    input = raw_input
except NameError:
    pass

project_name = 'pybind11'
project_dependencies = \
    [
        'waf-tools',
    ]


def importCode(code, name, add_to_sys_modules=0):
    """
    Import dynamically generated code as a module.
    Python recipe from http://code.activestate.com/recipes/82234
    """
    import imp

    module = imp.new_module(name)

    exec(code, module.__dict__)
    if add_to_sys_modules:
        sys.modules[name] = module

    return module


if __name__ == '__main__':
    print('Updating Smart Project Config Tool...')

    url = "https://raw.github.com/steinwurf/steinwurf-labs/" \
          "master/config_helper/config-impl.py"

    try:
        from urllib.request import urlopen, Request
    except ImportError:
        from urllib2 import urlopen, Request

    try:
        # Fetch the code file from the given url
        req = Request(url)
        response = urlopen(req)
        code = response.read()
        print("Update complete. Code size: {}\n".format(len(code)))
        try:
            # Import the code string as a module
            mod = importCode(code, "config_helper")
            # Run the actual config tool from the dynamic module
            mod.config_tool(project_dependencies, project_name)
        except:
            print("Unexpected error:")
            print(traceback.format_exc())
    except Exception as e:
        print("Could not fetch code file from:\n\t{}".format(url))
        print(e)

    input('Press ENTER to exit...')
