project('pybind11', 'cpp', default_options : ['cpp_std=c++14'], license : 'BSD-style')

py3_mod = import('python3')
py3_dep = dependency('python3')

pybind11_headers = files([
  'include/pybind11/detail/class.h',
  'include/pybind11/detail/common.h',
  'include/pybind11/detail/descr.h',
  'include/pybind11/detail/init.h',
  'include/pybind11/detail/internals.h',
  'include/pybind11/detail/typeid.h',
  'include/pybind11/attr.h',
  'include/pybind11/buffer_info.h',
  'include/pybind11/cast.h',
  'include/pybind11/chrono.h',
  'include/pybind11/common.h',
  'include/pybind11/complex.h',
  'include/pybind11/options.h',
  'include/pybind11/eigen.h',
  'include/pybind11/embed.h',
  'include/pybind11/eval.h',
  'include/pybind11/functional.h',
  'include/pybind11/numpy.h',
  'include/pybind11/operators.h',
  'include/pybind11/pybind11.h',
  'include/pybind11/pytypes.h',
  'include/pybind11/stl.h',
  'include/pybind11/stl_bind.h'
])


pybind11_incdir = include_directories('include')

pybind11_dep = declare_dependency(
	include_directories : pybind11_incdir,
	dependencies : py3_dep
)

subdir('tests')