from ._version import version_info, __version__

def get_include():
    """
    Return the directory that contains the pybind11 \\*.h header files.

    Extension modules that need to compile against pybind11 should use this
    function to locate the appropriate include directory.

    Notes
    -----
    When using ``distutils``, for example in ``setup.py``.
    ::
        import pybind11 as pb
        ...
        Extension('extension_name', ...
        include_dirs=[pb.get_include()])
        ...
    """
    import os
    return os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        '..', 'include'))
