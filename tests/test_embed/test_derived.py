# -*- coding: utf-8 -*-

import derived_module


def func():
    class Test(derived_module.test_derived):
        def func(self):
            return 42

    return Test()


def func2():
    class Test(derived_module.test_derived):
        pass

    return Test()
