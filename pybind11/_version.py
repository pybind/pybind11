# -*- coding: utf-8 -*-

import os


DIR = os.path.abspath(os.path.dirname(__file__))


def _to_int(s):
    try:
        return int(s)
    except ValueError:
        return s


# Get the version from the C++ file (in-source)
versions = {}

common_h = os.path.join(
    os.path.dirname(DIR), "include", "pybind11", "detail", "common.h"
)

with open(common_h) as f:
    for line in f:
        if "PYBIND11_VERSION_" in line:
            _, name, vers = line.split()
            versions[name[17:].lower()] = vers
            if len(versions) >= 3:
                break
    else:
        msg = "Version number not read correctly from {}: {}".format(common_h, versions)
        raise RuntimeError(msg)

__version__ = "{v[major]}.{v[minor]}.{v[patch]}".format(v=versions)
version_info = tuple(_to_int(s) for s in __version__.split("."))
