custom.so: test_module.cc
	 g++ test_module.cc -I/home/ethanid/miniconda3/envs/pybind11/include/python3.10 -I../include -shared -DNDEBUG -O3 -g -fPIC -std=c++14 -o custom.so

raw_custom.so: test_module_raw.cc
	 g++ test_module_raw.cc -I/home/ethanid/miniconda3/envs/pybind11/include/python3.10 -shared -DNDEBUG -O3 -fPIC -o raw_custom.so

raw_custom2.so: test_module_raw2.cc
	 g++ test_module_raw2.cc -I/home/ethanid/miniconda3/envs/pybind11/include/python3.10 -shared -DNDEBUG -O3 -fPIC -o raw_custom2.so


bench_results: custom.so raw_custom.so bench.py
	python bench.py > bench_results
