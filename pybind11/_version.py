# -*- coding: utf-8 -*-


def _to_int(s):
    try:
        return int(s)
    except ValueError:
        return s


__version__ = "2.6.1.dev1"
version_info = tuple(_to_int(s) for s in __version__.split("."))
