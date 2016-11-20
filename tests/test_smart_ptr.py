from pybind11_tests import ConstructorStats


def test_smart_ptr(capture):
    # Object1
    from pybind11_tests import (MyObject1, make_object_1, make_object_2,
                                print_object_1, print_object_2, print_object_3, print_object_4)

    for i, o in enumerate([make_object_1(), make_object_2(), MyObject1(3)], start=1):
        assert o.getRefCount() == 1
        with capture:
            print_object_1(o)
            print_object_2(o)
            print_object_3(o)
            print_object_4(o)
        assert capture == "MyObject1[{i}]\n".format(i=i) * 4

    from pybind11_tests import (make_myobject1_1, make_myobject1_2,
                                print_myobject1_1, print_myobject1_2,
                                print_myobject1_3, print_myobject1_4)

    for i, o in enumerate([make_myobject1_1(), make_myobject1_2(), MyObject1(6), 7], start=4):
        print(o)
        with capture:
            if not isinstance(o, int):
                print_object_1(o)
                print_object_2(o)
                print_object_3(o)
                print_object_4(o)
            print_myobject1_1(o)
            print_myobject1_2(o)
            print_myobject1_3(o)
            print_myobject1_4(o)
        assert capture == "MyObject1[{i}]\n".format(i=i) * (4 if isinstance(o, int) else 8)

    cstats = ConstructorStats.get(MyObject1)
    assert cstats.alive() == 0
    expected_values = ['MyObject1[{}]'.format(i) for i in range(1, 7)] + ['MyObject1[7]'] * 4
    assert cstats.values() == expected_values
    assert cstats.default_constructions == 0
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0 # Doesn't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0

    # Object2
    from pybind11_tests import (MyObject2, make_myobject2_1, make_myobject2_2,
                                make_myobject3_1, make_myobject3_2,
                                print_myobject2_1, print_myobject2_2,
                                print_myobject2_3, print_myobject2_4)

    for i, o in zip([8, 6, 7], [MyObject2(8), make_myobject2_1(), make_myobject2_2()]):
        print(o)
        with capture:
            print_myobject2_1(o)
            print_myobject2_2(o)
            print_myobject2_3(o)
            print_myobject2_4(o)
        assert capture == "MyObject2[{i}]\n".format(i=i) * 4

    cstats = ConstructorStats.get(MyObject2)
    assert cstats.alive() == 1
    o = None
    assert cstats.alive() == 0
    assert cstats.values() == ['MyObject2[8]', 'MyObject2[6]', 'MyObject2[7]']
    assert cstats.default_constructions == 0
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0 # Doesn't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0

    # Object3
    from pybind11_tests import (MyObject3, print_myobject3_1, print_myobject3_2,
                                print_myobject3_3, print_myobject3_4)

    for i, o in zip([9, 8, 9], [MyObject3(9), make_myobject3_1(), make_myobject3_2()]):
        print(o)
        with capture:
            print_myobject3_1(o)
            print_myobject3_2(o)
            print_myobject3_3(o)
            print_myobject3_4(o)
        assert capture == "MyObject3[{i}]\n".format(i=i) * 4

    cstats = ConstructorStats.get(MyObject3)
    assert cstats.alive() == 1
    o = None
    assert cstats.alive() == 0
    assert cstats.values() == ['MyObject3[9]', 'MyObject3[8]', 'MyObject3[9]']
    assert cstats.default_constructions == 0
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0 # Doesn't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0

    # Object and ref
    from pybind11_tests import Object, cstats_ref

    cstats = ConstructorStats.get(Object)
    assert cstats.alive() == 0
    assert cstats.values() == []
    assert cstats.default_constructions == 10
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0 # Doesn't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0

    cstats = cstats_ref()
    assert cstats.alive() == 0
    assert cstats.values() == ['from pointer'] * 10
    assert cstats.default_constructions == 30
    assert cstats.copy_constructions == 12
    # assert cstats.move_constructions >= 0 # Doesn't invoke any
    assert cstats.copy_assignments == 30
    assert cstats.move_assignments == 0


def test_unique_nodelete():
    from pybind11_tests import MyObject4
    o = MyObject4(23)
    assert o.value == 23
    cstats = ConstructorStats.get(MyObject4)
    assert cstats.alive() == 1
    del o
    cstats = ConstructorStats.get(MyObject4)
    assert cstats.alive() == 1  # Leak, but that's intentional
