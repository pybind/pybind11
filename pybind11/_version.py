from __future__ import annotations


def _to_int(s: str) -> int | str:
    try:
        return int(s)
    except ValueError:
        return s


__version__ = "2.13.1"
version_info = tuple(_to_int(s) for s in __version__.split("."))
