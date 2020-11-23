from pybind11_tests import unique_ptr_member as m


def test_pointee_and_ptr_owner():
    m.to_cout("")
    obj = m.pointee()
    assert obj.get_int() == 213
    del obj
    print("DONE.", flush=True)


test_pointee_and_ptr_owner()
