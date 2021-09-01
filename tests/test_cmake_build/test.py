# -*- coding: utf-8 -*-
import sys

import test_cmake_build

__file__  # Test this is properly set

assert test_cmake_build.add(1, 2) == 3
print("{} imports, runs, and adds: 1 + 2 = 3".format(sys.argv[1]))
