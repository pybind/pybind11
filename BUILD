# BUILD -- Build script for Bazel
#
# Copyright (c) 2016 Andreas Bergmeier <a.bergmeier@dsfishlabs.com>
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

cc_library(
	name = "pybind11",
	hdrs = glob([
		"include/pybind11/*.h",
	]),
	includes = [
		"include"
	],
	visibility = ["//visibility:public"],
)

