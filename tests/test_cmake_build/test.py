# -*- coding: utf-8 -*-
import sys

import test_cmake_build

if str is not bytes:  # If not Python2
    assert isinstance(__file__, str)  # Test this is properly set

assert test_cmake_build.add(1, 2) == 3
print("{} imports, runs, and adds: 1 + 2 = 3".format(sys.argv[1]))
