from typing import Union


def _to_int(s: str) -> Union[int, str]:
    try:
        return int(s)
    except ValueError:
        return s


__version__ = "2.11.1"
version_info = tuple(_to_int(s) for s in __version__.split("."))
