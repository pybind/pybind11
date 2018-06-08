from ._version import version_info, __version__  # noqa: F401 imported but unused


def get_include(user=False):
    from distutils.dist import Distribution
    import os
    import sys
    from os.path import abspath, dirname, join

    # Are we running in a virtual environment?
    virtualenv = hasattr(sys, 'real_prefix') or \
        sys.prefix != getattr(sys, "base_prefix", sys.prefix)

    if virtualenv:
        return os.path.join(sys.prefix, 'include', 'site',
                            'python' + sys.version[:3])
    else:
        dist = Distribution({'name': 'pybind11'})
        dist.parse_config_files()

        dist_cobj = dist.get_command_obj('install', create=True)

        # Search for packages in user's home directory?
        if user:
            dist_cobj.user = user
            dist_cobj.prefix = ""
        dist_cobj.finalize_options()

        libbase_suffix = dist_cobj.install_libbase.replace(dist_cobj.install_base, '').lstrip(os.path.sep)
        install_prefix = abspath(join(dirname(__file__), os.pardir).replace(libbase_suffix, ''))
        header_suffix = dirname(dist_cobj.install_headers).replace(dist_cobj.install_base, '').lstrip(os.path.sep)
        return join(install_prefix, header_suffix)
