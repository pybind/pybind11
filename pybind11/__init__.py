from ._version import version_info, __version__  # noqa: F401 imported but unused


def get_include(user=False):
    import os
    import sys
    from distutils.command.install import INSTALL_SCHEMES
    from site import USER_BASE, USER_SITE

    d = os.path.dirname(__file__)

    expand_list = [("$py_version_nodot", "%d%d" % sys.version_info[:2]),
                   ("$py_version_short", "%d.%d" % sys.version_info[:2]),
                   ("$abiflags", getattr(sys, "abiflags", "")),
                   ("$usersite", USER_SITE),
                   ("$userbase", USER_BASE),
                   ("$dist_name", "pybind11")]

    def expand_path(_path):
        for _from, _to in expand_list:
            _path = _path.replace(_from, _to)
        return _path

    purelib = os.path.dirname(d)

    for v in INSTALL_SCHEMES.values():
        relpath = os.path.relpath(expand_path(v["headers"]),
                                  start=expand_path(v["purelib"]))
        headers = os.path.realpath(os.path.join(purelib, relpath))
        if os.path.exists(headers):
            # Package is installed
            return os.path.dirname(headers)
    else:
        # Package is from a source directory
        return os.path.join(os.path.dirname(d), "include")
