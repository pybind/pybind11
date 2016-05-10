To release a new version of pybind11:

- Update the version number and push to pypi
    - Update ``pybind11/_version.py`` (set release version, remove 'dev')
    - Tag release date in ``doc/changelog.rst``.
    - ``git add`` and ``git commit``.
    - ``git tag -a vX.Y -m 'vX.Y release'``.
    - ``git push``
    - ``git push --tags``.
    - ``python setup.py sdist upload``.
    - ``python setup.py bdist_wheel upload``.
- Update conda-forge (https://github.com/conda-forge/pybind11-feedstock)
    - change version number in ``meta.yml``
    - update checksum to match the one computed by pypi
- Get back to work
    - Update ``_version.py`` (add 'dev' and increment minor).
    - Update version macros in ``include/pybind11/common.h``
    - ``git add`` and ``git commit``.
      ``git push``
