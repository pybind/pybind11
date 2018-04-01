
catch_dep = dependency('catch', required:true)
thread_dep = dependency('threads')

external_module_mod = py3_mod.extension_module('external_module',
	sources: 'external_module.cpp',
	dependencies: pybind11_dep)

test_embed = executable('test_embed',
			sources:['test_interpreter.cpp', 'catch.cpp'],
			dependencies:[catch_dep, thread_dep, pybind11_dep]
			)

test('test_embed', test_embed,
		env:'PYTHONPATH='+meson.current_build_dir(),
	  	workdir:meson.current_source_dir())