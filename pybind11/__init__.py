# -*- coding: utf-8 -*-
from ._version import version_info, __version__  # noqa: F401 imported but unused


def get_include(user=False):
    import os
    d = os.path.dirname(__file__)
    if os.path.exists(os.path.join(d, "include")):
        # Package is installed
        return os.path.join(d, "include")
    else:
        # Package is from a source directory
        return os.path.join(os.path.dirname(d), "include")
