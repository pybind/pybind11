To release a new version of pybind11:

- Update the version number and push to pypi
    - Update ``pybind11/_version.py`` (set release version, remove 'dev').
    - Update ``PYBIND11_VERSION_MAJOR`` etc. in ``include/pybind11/detail/common.h``.
    - Ensure that all the information in ``setup.py`` is up-to-date.
    - Update version in ``docs/conf.py``.
    - Tag release date in ``docs/changelog.rst``.
    - ``git add`` and ``git commit``.
    - if new minor version: ``git checkout -b vX.Y``, ``git push -u origin vX.Y``
    - ``git tag -a vX.Y.Z -m 'vX.Y.Z release'``.
    - ``git push``
    - ``git push --tags``.
    - ``python setup.py sdist upload``.
    - ``python setup.py bdist_wheel upload``.
- Get back to work
    - Update ``_version.py`` (add 'dev' and increment minor).
    - Update version in ``docs/conf.py``
    - Update version macros in ``include/pybind11/common.h``
    - ``git add`` and ``git commit``.
      ``git push``
