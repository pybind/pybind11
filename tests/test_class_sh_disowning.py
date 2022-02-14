import pytest

from pybind11_tests import class_sh_disowning as m


def test_same_twice():
    while True:
        obj1a = m.Atype1(57)
        obj1b = m.Atype1(62)
        assert m.same_twice(obj1a, obj1b) == (57 * 10 + 1) * 100 + (62 * 10 + 1) * 10
        obj1c = m.Atype1(0)
        with pytest.raises(ValueError):
            # Disowning works for one argument, but not both.
            m.same_twice(obj1c, obj1c)
        with pytest.raises(ValueError):
            obj1c.get()
        return  # Comment out for manual leak checking (use `top` command).


def test_mixed():
    first_pass = True
    while True:
        obj1a = m.Atype1(90)
        obj2a = m.Atype2(25)
        assert m.mixed(obj1a, obj2a) == (90 * 10 + 1) * 200 + (25 * 10 + 2) * 20

        # The C++ order of evaluation of function arguments is (unfortunately) unspecified:
        # https://en.cppreference.com/w/cpp/language/eval_order
        # Read on.
        obj1b = m.Atype1(0)
        with pytest.raises(ValueError):
            # If the 1st argument is evaluated first, obj1b is disowned before the conversion for
            # the already disowned obj2a fails as expected.
            m.mixed(obj1b, obj2a)
        obj2b = m.Atype2(0)
        with pytest.raises(ValueError):
            # If the 2nd argument is evaluated first, obj2b is disowned before the conversion for
            # the already disowned obj1a fails as expected.
            m.mixed(obj1a, obj2b)

        def is_disowned(obj):
            try:
                obj.get()
            except ValueError:
                return True
            return False

        # Either obj1b or obj2b was disowned in the expected failed m.mixed() calls above, but not
        # both.
        is_disowned_results = (is_disowned(obj1b), is_disowned(obj2b))
        assert is_disowned_results.count(True) == 1
        if first_pass:
            first_pass = False
            print(
                "\nC++ function argument %d is evaluated first."
                % (is_disowned_results.index(True) + 1)
            )

        return  # Comment out for manual leak checking (use `top` command).


def test_overloaded():
    while True:
        obj1 = m.Atype1(81)
        obj2 = m.Atype2(60)
        with pytest.raises(TypeError):
            m.overloaded(obj1, "NotInt")
        assert obj1.get() == 81 * 10 + 1  # Not disowned.
        assert m.overloaded(obj1, 3) == (81 * 10 + 1) * 30 + 3
        with pytest.raises(TypeError):
            m.overloaded(obj2, "NotInt")
        assert obj2.get() == 60 * 10 + 2  # Not disowned.
        assert m.overloaded(obj2, 2) == (60 * 10 + 2) * 40 + 2
        return  # Comment out for manual leak checking (use `top` command).
