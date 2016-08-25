

def test_chrono_system_clock():
    from pybind11_tests import test_chrono1
    import datetime

    # Get the time from both c++ and datetime
    date1 = test_chrono1()
    date2 = datetime.datetime.today()

    # The returned value should be a datetime
    assert isinstance(date1, datetime.datetime)

    # The numbers should vary by a very small amount (time it took to execute)
    diff = abs(date1 - date2)

    # There should never be a days/seconds difference
    assert diff.days == 0
    assert diff.seconds == 0

    # 50 milliseconds is a very long time to execute this
    assert diff.microseconds < 50000


def test_chrono_system_clock_roundtrip():
    from pybind11_tests import test_chrono2
    import datetime

    date1 = datetime.datetime.today()

    # Roundtrip the time
    date2 = test_chrono2(date1)

    # The returned value should be a datetime
    assert isinstance(date2, datetime.datetime)

    # They should be identical (no information lost on roundtrip)
    diff = abs(date1 - date2)
    assert diff.days == 0
    assert diff.seconds == 0
    assert diff.microseconds == 0


def test_chrono_duration_roundtrip():
    from pybind11_tests import test_chrono3
    import datetime

    # Get the difference betwen two times (a timedelta)
    date1 = datetime.datetime.today()
    date2 = datetime.datetime.today()
    diff = date2 - date1

    # Make sure this is a timedelta
    assert isinstance(diff, datetime.timedelta)

    cpp_diff = test_chrono3(diff)

    assert cpp_diff.days == diff.days
    assert cpp_diff.seconds == diff.seconds
    assert cpp_diff.microseconds == diff.microseconds


def test_chrono_duration_subtraction_equivalence():
    from pybind11_tests import test_chrono4
    import datetime

    date1 = datetime.datetime.today()
    date2 = datetime.datetime.today()

    diff = date2 - date1
    cpp_diff = test_chrono4(date2, date1)

    assert cpp_diff.days == diff.days
    assert cpp_diff.seconds == diff.seconds
    assert cpp_diff.microseconds == diff.microseconds


def test_chrono_steady_clock():
    from pybind11_tests import test_chrono5
    import datetime

    time1 = test_chrono5()
    time2 = test_chrono5()

    assert isinstance(time1, datetime.time)
    assert isinstance(time2, datetime.time)


def test_chrono_steady_clock_roundtrip():
    from pybind11_tests import test_chrono6
    import datetime

    time1 = datetime.time(second=10, microsecond=100)
    time2 = test_chrono6(time1)

    assert isinstance(time2, datetime.time)

    # They should be identical (no information lost on roundtrip)
    assert time1.hour == time2.hour
    assert time1.minute == time2.minute
    assert time1.second == time2.second
    assert time1.microsecond == time2.microsecond
