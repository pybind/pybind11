To release a new version of pybind11:

- Update version macros in `include/pybind11/common.h`
- Update `pybind11/_version.py` (set release version, remove 'dev')
- `git add` and `git commit`.
- `python setup.py sdist upload`.
- `python setup.py bdist_wheel upload`.
- `git tag -a X.X -m 'Release tag comment'`.
- Update `_version.py` (add 'dev' and increment minor).
- `git add` and `git commit`. `git push`. `git push --tags`.

The remote for the last `git push --tags` should be the main repository for
pybind11.
