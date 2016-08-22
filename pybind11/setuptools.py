from __future__ import absolute_import

import distutils.ccompiler
import setuptools
import sys
import tempfile

import pybind11


class Extension(setuptools.Extension, object):
    _extra_compile_args = None

    def __init__(self, name, sources, **kwargs):
        if self._extra_compile_args is None:
            self._set_extra_compile_args()
        kwargs["language"] = "c++"
        # We append the include dirs but prepend the extra compile args to make
        # them overridable.
        kwargs.setdefault("include_dirs", []).extend(
            [pybind11.get_include(),
             pybind11.get_include(user=True)])
        kwargs.setdefault("extra_compile_args", [])[:0] = (
            self._extra_compile_args)
        super(Extension, self).__init__(name, sources, **kwargs)

    @classmethod
    def _set_extra_compile_args(cls):
        compiler = distutils.ccompiler.new_compiler()
        ct = compiler.compiler_type
        if ct == 'unix':
            opts = []
            if sys.platform == 'darwin':
                opts += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
            opts += [cpp_flag(compiler)]
            if has_flag(compiler, '-fvisibility=hidden'):
                opts += ['-fvisibility=hidden']
        elif ct == 'msvc':
            opts = ['/EHsc']
        cls._extra_compile_args = opts


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    from shutil import rmtree  # Ensure it remains available at shutdown.
    tmpdir = tempfile.mkdtemp()
    try:
        with tempfile.NamedTemporaryFile('w', suffix='.cpp', dir=tmpdir) as f:
            f.write('int main (int argc, char **argv) { return 0; }')
            try:
                compiler.compile(
                    [f.name], output_dir=tmpdir, extra_postargs=[flagname])
            except setuptools.distutils.errors.CompileError:
                return False
        return True
    finally:
        rmtree(tmpdir)


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')
