from ._version import version_info, __version__


def get_include():
    import os
    try:
        from pip import locations
        return os.path.dirname(
            locations.distutils_scheme('pybind11')['headers'])
    except ImportError:
        return 'include'
