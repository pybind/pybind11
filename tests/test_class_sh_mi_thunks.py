import pytest

from pybind11_tests import class_sh_mi_thunks as m


@pytest.mark.parametrize(
    "vec_size_fn",
    [
        m.vec_size_base0_raw_ptr,
        m.vec_size_base0_shared_ptr,
        m.vec_size_base0_unique_ptr,
    ],
)
@pytest.mark.parametrize(
    "get_fn",
    [
        m.get_derived_as_base0_raw_ptr,
        m.get_derived_as_base0_shared_ptr,
        m.get_derived_as_base0_unique_ptr,
    ],
)
def test_get_vec_size(get_fn, vec_size_fn):
    obj = get_fn()
    if (
        get_fn is m.get_derived_as_base0_shared_ptr
        and vec_size_fn is m.vec_size_base0_unique_ptr
    ):
        with pytest.raises(ValueError) as exc_info:
            vec_size_fn(obj)
        assert (
            str(exc_info.value)
            == "Cannot disown external shared_ptr (loaded_as_unique_ptr)."
        )
    else:
        assert vec_size_fn(obj) == 5
        if vec_size_fn is m.vec_size_base0_unique_ptr:
            with pytest.raises(ValueError) as exc_info:
                vec_size_fn(obj)
            assert (
                str(exc_info.value)
                == "Missing value for wrapped C++ type: Python instance was disowned."
            )
