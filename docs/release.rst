To release a new version of pybind11:

- Update the version number and push to pypi
    - Update ``pybind11/_version.py`` (set release version, remove 'dev')
    - ``git add`` and ``git commit``.
    - ``python setup.py sdist upload``.
    - ``python setup.py bdist_wheel upload``.
- Tag the commit and push to anaconda.org
    - ``git tag -a X.X -m '[Release comment]'``.
    - ``conda-build conda.recipe --output``
      This should ouput the path of the generated tar.bz2 for the package
    - ``conda-convert --platform all [path/to/tar.bz2] -o .``
    - For each platform name, ``anaconda-upload -u pybind ./platform/tarfile.tar.bz2``
- Get back to work
    - Update ``_version.py`` (add 'dev' and increment minor).
    - Update version macros in ``include/pybind11/common.h``
    - ``git add`` and ``git commit``. ``git push``. ``git push --tags``.

The remote for the last ``git push --tags`` should be the main repository for
pybind11.
