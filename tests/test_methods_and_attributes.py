from pybind11_tests import ExampleMandA, ConstructorStats


def test_methods_and_attributes():
    instance1 = ExampleMandA()
    instance2 = ExampleMandA(32)

    instance1.add1(instance2)
    instance1.add2(instance2)
    instance1.add3(instance2)
    instance1.add4(instance2)
    instance1.add5(instance2)
    instance1.add6(32)
    instance1.add7(32)
    instance1.add8(32)
    instance1.add9(32)
    instance1.add10(32)

    assert str(instance1) == "ExampleMandA[value=320]"
    assert str(instance2) == "ExampleMandA[value=32]"
    assert str(instance1.self1()) == "ExampleMandA[value=320]"
    assert str(instance1.self2()) == "ExampleMandA[value=320]"
    assert str(instance1.self3()) == "ExampleMandA[value=320]"
    assert str(instance1.self4()) == "ExampleMandA[value=320]"
    assert str(instance1.self5()) == "ExampleMandA[value=320]"

    assert instance1.internal1() == 320
    assert instance1.internal2() == 320
    assert instance1.internal3() == 320
    assert instance1.internal4() == 320
    assert instance1.internal5() == 320

    assert instance1.value == 320
    instance1.value = 100
    assert str(instance1) == "ExampleMandA[value=100]"

    cstats = ConstructorStats.get(ExampleMandA)
    assert cstats.alive() == 2
    del instance1, instance2
    assert cstats.alive() == 0
    assert cstats.values() == ["32"]
    assert cstats.default_constructions == 1
    assert cstats.copy_constructions == 3
    assert cstats.move_constructions >= 1
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0
