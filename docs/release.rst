To release a new version of pybind11:

- Update version macros in `include/pybind11/common.h`
- Update version in `setup.py`
- `git add` and `git commit`.
- `python setup.py sdist upload`.
- `python setup.py bdist_wheel upload`.
- `git tag -a X.X -m 'Release tag comment'`.
- `git add` and `git commit`. `git push`. `git push --tags`.

The remote for the last `git push --tags` should be the main repository for
pybind11.
