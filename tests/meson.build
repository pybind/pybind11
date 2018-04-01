py3_exe = find_program('python3', required:true)
pytest = find_program('pytest-3', required:true)

pybind11_tests = [
  'test_buffers',
  'test_builtin_casters',
  'test_call_policies',
  'test_callbacks',
  'test_chrono',
  'test_class',
  'test_constants_and_functions',
  'test_copy_move',
  'test_docstring_options',
  'test_enum',
  'test_eval',
  'test_exceptions',
  'test_factory_constructors',
  'test_iostream',
  'test_kwargs_and_defaults',
  'test_local_bindings',
  'test_methods_and_attributes',
  'test_modules',
  'test_multiple_inheritance',
  'test_numpy_array',
  'test_numpy_dtypes',
  'test_numpy_vectorize',
  'test_opaque_types',
  'test_operator_overloading',
  'test_pickling',
  'test_pytypes',
  'test_sequences_and_iterators',
  'test_smart_ptr',
  'test_stl',
  'test_stl_binders',
  'test_virtual_functions'
]

pybind11_tests_sources = ['pybind11_tests.cpp']
pybind11_tests_pysources = []
pybind11_tests_deps = [pybind11_dep]

foreach test_name : pybind11_tests
	pybind11_tests_sources += test_name+'.cpp'
	pybind11_tests_pysources += test_name+'.py'
endforeach

eigen3_dep = dependency('eigen3', required:false)

if eigen3_dep.found()
	pybind11_tests_sources += 'test_eigen.cpp'
	pybind11_tests_pysources += 'test_eigen.py'
	pybind11_tests_deps += eigen3_dep
endif

pybind11_test_mod = py3_mod.extension_module('pybind11_tests',
	sources: pybind11_tests_sources,
	dependencies: pybind11_tests_deps)

pybind11_cross_module_tests = py3_mod.extension_module('pybind11_cross_module_tests',
	sources: 'pybind11_cross_module_tests.cpp',
	dependencies: pybind11_dep)

foreach test_name : pybind11_tests
	test(test_name, py3_exe,
		args:['-m','pytest', test_name+'.py'],
		env:'PYTHONPATH='+meson.current_build_dir(),
	  	workdir:meson.current_source_dir()
	  	)
endforeach

subdir('test_embed')