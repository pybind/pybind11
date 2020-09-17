import ctypes

from pybind11_tests import ctypes_buffer as m

def test_ctypes_buffer():
    assert m.get_ctypes_buffer_size((ctypes.c_char * 10)()) == 10

if __name__ == "__main__":
    test_ctypes_buffer()
