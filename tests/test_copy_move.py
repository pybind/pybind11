import pytest
from pybind11_tests import has_optional


def test_lacking_copy_ctor():
    from pybind11_tests import lacking_copy_ctor
    with pytest.raises(RuntimeError) as excinfo:
        lacking_copy_ctor.get_one()
    assert "the object is non-copyable!" in str(excinfo.value)


def test_lacking_move_ctor():
    from pybind11_tests import lacking_move_ctor
    with pytest.raises(RuntimeError) as excinfo:
        lacking_move_ctor.get_one()
    assert "the object is neither movable nor copyable!" in str(excinfo.value)


def test_move_and_copy_casts():
    """Cast some values in C++ via custom type casters and count the number of moves/copies."""
    from pybind11_tests import move_and_copy_casts, move_and_copy_cstats

    cstats = move_and_copy_cstats()
    c_m, c_mc, c_c = cstats["MoveOnlyInt"], cstats["MoveOrCopyInt"], cstats["CopyOnlyInt"]

    # The type move constructions/assignments below each get incremented: the move assignment comes
    # from the type_caster load; the move construction happens when extracting that via a cast or
    # loading into an argument.
    assert move_and_copy_casts(3) == 18
    assert c_m.copy_assignments + c_m.copy_constructions == 0
    assert c_m.move_assignments == 2
    assert c_m.move_constructions == 2
    assert c_mc.alive() == 0
    assert c_mc.copy_assignments + c_mc.copy_constructions == 0
    assert c_mc.move_assignments == 2
    assert c_mc.move_constructions == 2
    assert c_c.alive() == 0
    assert c_c.copy_assignments == 2
    assert c_c.copy_constructions == 2
    assert c_m.alive() + c_mc.alive() + c_c.alive() == 0


def test_move_and_copy_loads():
    """Call some functions that load arguments via custom type casters and count the number of
    moves/copies."""
    from pybind11_tests import (move_and_copy_cstats, move_only, move_or_copy, copy_only,
                                move_pair, move_tuple, copy_tuple, move_copy_nested)

    cstats = move_and_copy_cstats()
    c_m, c_mc, c_c = cstats["MoveOnlyInt"], cstats["MoveOrCopyInt"], cstats["CopyOnlyInt"]

    assert move_only(10) == 10  # 1 move, c_m
    assert move_or_copy(11) == 11  # 1 move, c_mc
    assert copy_only(12) == 12  # 1 copy, c_c
    assert move_pair((13, 14)) == 27  # 1 c_m move, 1 c_mc move
    assert move_tuple((15, 16, 17)) == 48  # 2 c_m moves, 1 c_mc move
    assert copy_tuple((18, 19)) == 37  # 2 c_c copies
    # Direct constructions: 2 c_m moves, 2 c_mc moves, 1 c_c copy
    # Extra moves/copies when moving pairs/tuples: 3 c_m, 3 c_mc, 2 c_c
    assert move_copy_nested((1, ((2, 3, (4,)), 5))) == 15

    assert c_m.copy_assignments + c_m.copy_constructions == 0
    assert c_m.move_assignments == 6
    assert c_m.move_constructions == 9
    assert c_mc.copy_assignments + c_mc.copy_constructions == 0
    assert c_mc.move_assignments == 5
    assert c_mc.move_constructions == 8
    assert c_c.copy_assignments == 4
    assert c_c.copy_constructions == 6
    assert c_m.alive() + c_mc.alive() + c_c.alive() == 0


@pytest.mark.skipif(not has_optional, reason='no <optional>')
def test_move_and_copy_load_optional():
    """Tests move/copy loads of std::optional arguments"""
    from pybind11_tests import (move_and_copy_cstats, move_optional, move_or_copy_optional,
                                copy_optional, move_optional_tuple)

    cstats = move_and_copy_cstats()
    c_m, c_mc, c_c = cstats["MoveOnlyInt"], cstats["MoveOrCopyInt"], cstats["CopyOnlyInt"]

    # The extra move/copy constructions below come from the std::optional move (which has to move
    # its arguments):
    assert move_optional(10) == 10  # c_m: 1 move assign, 2 move construct
    assert move_or_copy_optional(11) == 11  # c_mc: 1 move assign, 2 move construct
    assert copy_optional(12) == 12  # c_c: 1 copy assign, 2 copy construct
    # 1 move assign + move construct moves each of c_m, c_mc, 1 c_c copy
    # +1 move/copy construct each from moving the tuple
    # +1 move/copy construct each from moving the optional (which moves the tuple again)
    assert move_optional_tuple((3, 4, 5)) == 12

    assert c_m.copy_assignments + c_m.copy_constructions == 0
    assert c_m.move_assignments == 2
    assert c_m.move_constructions == 5
    assert c_mc.copy_assignments + c_mc.copy_constructions == 0
    assert c_mc.move_assignments == 2
    assert c_mc.move_constructions == 5
    assert c_c.copy_assignments == 2
    assert c_c.copy_constructions == 5
    assert c_m.alive() + c_mc.alive() + c_c.alive() == 0
