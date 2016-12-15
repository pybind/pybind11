import pytest
import pybind11_tests
from pybind11_tests import ConstructorStats


def test_override(capture, msg):
    from pybind11_tests import (ExampleVirt, runExampleVirt, runExampleVirtVirtual,
                                runExampleVirtBool)

    class ExtendedExampleVirt(ExampleVirt):
        def __init__(self, state):
            super(ExtendedExampleVirt, self).__init__(state + 1)
            self.data = "Hello world"

        def run(self, value):
            print('ExtendedExampleVirt::run(%i), calling parent..' % value)
            return super(ExtendedExampleVirt, self).run(value + 1)

        def run_bool(self):
            print('ExtendedExampleVirt::run_bool()')
            return False

        def get_string1(self):
            return "override1"

        def pure_virtual(self):
            print('ExtendedExampleVirt::pure_virtual(): %s' % self.data)

    class ExtendedExampleVirt2(ExtendedExampleVirt):
        def __init__(self, state):
            super(ExtendedExampleVirt2, self).__init__(state + 1)

        def get_string2(self):
            return "override2"

    ex12 = ExampleVirt(10)
    with capture:
        assert runExampleVirt(ex12, 20) == 30
    assert capture == """
        Original implementation of ExampleVirt::run(state=10, value=20, str1=default1, str2=default2)
    """  # noqa: E501 line too long

    with pytest.raises(RuntimeError) as excinfo:
        runExampleVirtVirtual(ex12)
    assert msg(excinfo.value) == 'Tried to call pure virtual function "ExampleVirt::pure_virtual"'

    ex12p = ExtendedExampleVirt(10)
    with capture:
        assert runExampleVirt(ex12p, 20) == 32
    assert capture == """
        ExtendedExampleVirt::run(20), calling parent..
        Original implementation of ExampleVirt::run(state=11, value=21, str1=override1, str2=default2)
    """  # noqa: E501 line too long
    with capture:
        assert runExampleVirtBool(ex12p) is False
    assert capture == "ExtendedExampleVirt::run_bool()"
    with capture:
        runExampleVirtVirtual(ex12p)
    assert capture == "ExtendedExampleVirt::pure_virtual(): Hello world"

    ex12p2 = ExtendedExampleVirt2(15)
    with capture:
        assert runExampleVirt(ex12p2, 50) == 68
    assert capture == """
        ExtendedExampleVirt::run(50), calling parent..
        Original implementation of ExampleVirt::run(state=17, value=51, str1=override1, str2=override2)
    """  # noqa: E501 line too long

    cstats = ConstructorStats.get(ExampleVirt)
    assert cstats.alive() == 3
    del ex12, ex12p, ex12p2
    assert cstats.alive() == 0
    assert cstats.values() == ['10', '11', '17']
    assert cstats.copy_constructions == 0
    assert cstats.move_constructions >= 0


def test_inheriting_repeat():
    from pybind11_tests import A_Repeat, B_Repeat, C_Repeat, D_Repeat, A_Tpl, B_Tpl, C_Tpl, D_Tpl

    class AR(A_Repeat):
        def unlucky_number(self):
            return 99

    class AT(A_Tpl):
        def unlucky_number(self):
            return 999

    obj = AR()
    assert obj.say_something(3) == "hihihi"
    assert obj.unlucky_number() == 99
    assert obj.say_everything() == "hi 99"

    obj = AT()
    assert obj.say_something(3) == "hihihi"
    assert obj.unlucky_number() == 999
    assert obj.say_everything() == "hi 999"

    for obj in [B_Repeat(), B_Tpl()]:
        assert obj.say_something(3) == "B says hi 3 times"
        assert obj.unlucky_number() == 13
        assert obj.lucky_number() == 7.0
        assert obj.say_everything() == "B says hi 1 times 13"

    for obj in [C_Repeat(), C_Tpl()]:
        assert obj.say_something(3) == "B says hi 3 times"
        assert obj.unlucky_number() == 4444
        assert obj.lucky_number() == 888.0
        assert obj.say_everything() == "B says hi 1 times 4444"

    class CR(C_Repeat):
        def lucky_number(self):
            return C_Repeat.lucky_number(self) + 1.25

    obj = CR()
    assert obj.say_something(3) == "B says hi 3 times"
    assert obj.unlucky_number() == 4444
    assert obj.lucky_number() == 889.25
    assert obj.say_everything() == "B says hi 1 times 4444"

    class CT(C_Tpl):
        pass

    obj = CT()
    assert obj.say_something(3) == "B says hi 3 times"
    assert obj.unlucky_number() == 4444
    assert obj.lucky_number() == 888.0
    assert obj.say_everything() == "B says hi 1 times 4444"

    class CCR(CR):
        def lucky_number(self):
            return CR.lucky_number(self) * 10

    obj = CCR()
    assert obj.say_something(3) == "B says hi 3 times"
    assert obj.unlucky_number() == 4444
    assert obj.lucky_number() == 8892.5
    assert obj.say_everything() == "B says hi 1 times 4444"

    class CCT(CT):
        def lucky_number(self):
            return CT.lucky_number(self) * 1000

    obj = CCT()
    assert obj.say_something(3) == "B says hi 3 times"
    assert obj.unlucky_number() == 4444
    assert obj.lucky_number() == 888000.0
    assert obj.say_everything() == "B says hi 1 times 4444"

    class DR(D_Repeat):
        def unlucky_number(self):
            return 123

        def lucky_number(self):
            return 42.0

    for obj in [D_Repeat(), D_Tpl()]:
        assert obj.say_something(3) == "B says hi 3 times"
        assert obj.unlucky_number() == 4444
        assert obj.lucky_number() == 888.0
        assert obj.say_everything() == "B says hi 1 times 4444"

    obj = DR()
    assert obj.say_something(3) == "B says hi 3 times"
    assert obj.unlucky_number() == 123
    assert obj.lucky_number() == 42.0
    assert obj.say_everything() == "B says hi 1 times 123"

    class DT(D_Tpl):
        def say_something(self, times):
            return "DT says:" + (' quack' * times)

        def unlucky_number(self):
            return 1234

        def lucky_number(self):
            return -4.25

    obj = DT()
    assert obj.say_something(3) == "DT says: quack quack quack"
    assert obj.unlucky_number() == 1234
    assert obj.lucky_number() == -4.25
    assert obj.say_everything() == "DT says: quack 1234"

    class DT2(DT):
        def say_something(self, times):
            return "DT2: " + ('QUACK' * times)

        def unlucky_number(self):
            return -3

    class BT(B_Tpl):
        def say_something(self, times):
            return "BT" * times

        def unlucky_number(self):
            return -7

        def lucky_number(self):
            return -1.375

    obj = BT()
    assert obj.say_something(3) == "BTBTBT"
    assert obj.unlucky_number() == -7
    assert obj.lucky_number() == -1.375
    assert obj.say_everything() == "BT -7"


# PyPy: Reference count > 1 causes call with noncopyable instance
# to fail in ncv1.print_nc()
@pytest.unsupported_on_pypy
@pytest.mark.skipif(not hasattr(pybind11_tests, 'NCVirt'),
                    reason="NCVirt test broken on ICPC")
def test_move_support():
    from pybind11_tests import NCVirt, NonCopyable, Movable

    class NCVirtExt(NCVirt):
        def get_noncopyable(self, a, b):
            # Constructs and returns a new instance:
            nc = NonCopyable(a * a, b * b)
            return nc

        def get_movable(self, a, b):
            # Return a referenced copy
            self.movable = Movable(a, b)
            return self.movable

    class NCVirtExt2(NCVirt):
        def get_noncopyable(self, a, b):
            # Keep a reference: this is going to throw an exception
            self.nc = NonCopyable(a, b)
            return self.nc

        def get_movable(self, a, b):
            # Return a new instance without storing it
            return Movable(a, b)

    ncv1 = NCVirtExt()
    assert ncv1.print_nc(2, 3) == "36"
    assert ncv1.print_movable(4, 5) == "9"
    ncv2 = NCVirtExt2()
    assert ncv2.print_movable(7, 7) == "14"
    # Don't check the exception message here because it differs under debug/non-debug mode
    with pytest.raises(RuntimeError):
        ncv2.print_nc(9, 9)

    nc_stats = ConstructorStats.get(NonCopyable)
    mv_stats = ConstructorStats.get(Movable)
    assert nc_stats.alive() == 1
    assert mv_stats.alive() == 1
    del ncv1, ncv2
    assert nc_stats.alive() == 0
    assert mv_stats.alive() == 0
    assert nc_stats.values() == ['4', '9', '9', '9']
    assert mv_stats.values() == ['4', '5', '7', '7']
    assert nc_stats.copy_constructions == 0
    assert mv_stats.copy_constructions == 1
    assert nc_stats.move_constructions >= 0
    assert mv_stats.move_constructions >= 0
