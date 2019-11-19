from ._version import version_info, __version__  # noqa: F401 imported but unused


def get_include(user=False):
    import os
    d = os.path.dirname(__file__)
    return os.path.join(os.path.dirname(d), "include")
