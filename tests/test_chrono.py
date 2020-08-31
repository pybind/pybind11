# -*- coding: utf-8 -*-
from pybind11_tests import chrono as m
import datetime
import pytest

import env  # noqa: F401


def test_chrono_system_clock():

    # Get the time from both c++ and datetime
    date0 = datetime.datetime.today()
    date1 = m.test_chrono1()
    date2 = datetime.datetime.today()

    # The returned value should be a datetime
    assert isinstance(date1, datetime.datetime)

    # The numbers should vary by a very small amount (time it took to execute)
    diff_python = abs(date2 - date0)
    diff = abs(date1 - date2)

    # There should never be a days difference
    assert diff.days == 0

    # Since datetime.datetime.today() calls time.time(), and on some platforms
    # that has 1 second accuracy, we compare this way
    assert diff.seconds <= diff_python.seconds


def test_chrono_system_clock_roundtrip():
    date1 = datetime.datetime.today()

    # Roundtrip the time
    date2 = m.test_chrono2(date1)

    # The returned value should be a datetime
    assert isinstance(date2, datetime.datetime)

    # They should be identical (no information lost on roundtrip)
    diff = abs(date1 - date2)
    assert diff.days == 0
    assert diff.seconds == 0
    assert diff.microseconds == 0


def test_chrono_system_clock_roundtrip_date():
    date1 = datetime.date.today()

    # Roundtrip the time
    datetime2 = m.test_chrono2(date1)
    date2 = datetime2.date()
    time2 = datetime2.time()

    # The returned value should be a datetime
    assert isinstance(datetime2, datetime.datetime)
    assert isinstance(date2, datetime.date)
    assert isinstance(time2, datetime.time)

    # They should be identical (no information lost on roundtrip)
    diff = abs(date1 - date2)
    assert diff.days == 0
    assert diff.seconds == 0
    assert diff.microseconds == 0

    # Year, Month & Day should be the same after the round trip
    assert date1.year == date2.year
    assert date1.month == date2.month
    assert date1.day == date2.day

    # There should be no time information
    assert time2.hour == 0
    assert time2.minute == 0
    assert time2.second == 0
    assert time2.microsecond == 0


SKIP_TZ_ENV_ON_WIN = pytest.mark.skipif(
    "env.WIN", reason="TZ environment variable only supported on POSIX"
)


@pytest.mark.parametrize("time1", [
    datetime.datetime.today().time(),
    datetime.time(0, 0, 0),
    datetime.time(0, 0, 0, 1),
    datetime.time(0, 28, 45, 109827),
    datetime.time(0, 59, 59, 999999),
    datetime.time(1, 0, 0),
    datetime.time(5, 59, 59, 0),
    datetime.time(5, 59, 59, 1),
])
@pytest.mark.parametrize("tz", [
    None,
    pytest.param("Europe/Brussels", marks=SKIP_TZ_ENV_ON_WIN),
    pytest.param("Asia/Pyongyang", marks=SKIP_TZ_ENV_ON_WIN),
    pytest.param("America/New_York", marks=SKIP_TZ_ENV_ON_WIN),
])
def test_chrono_system_clock_roundtrip_time(time1, tz, monkeypatch):
    if tz is not None:
        monkeypatch.setenv("TZ", "/usr/share/zoneinfo/{}".format(tz))

    # Roundtrip the time
    datetime2 = m.test_chrono2(time1)
    date2 = datetime2.date()
    time2 = datetime2.time()

    # The returned value should be a datetime
    assert isinstance(datetime2, datetime.datetime)
    assert isinstance(date2, datetime.date)
    assert isinstance(time2, datetime.time)

    # Hour, Minute, Second & Microsecond should be the same after the round trip
    assert time1.hour == time2.hour
    assert time1.minute == time2.minute
    assert time1.second == time2.second
    assert time1.microsecond == time2.microsecond

    # There should be no date information (i.e. date = python base date)
    assert date2.year == 1970
    assert date2.month == 1
    assert date2.day == 1


def test_chrono_duration_roundtrip():

    # Get the difference between two times (a timedelta)
    date1 = datetime.datetime.today()
    date2 = datetime.datetime.today()
    diff = date2 - date1

    # Make sure this is a timedelta
    assert isinstance(diff, datetime.timedelta)

    cpp_diff = m.test_chrono3(diff)

    assert cpp_diff.days == diff.days
    assert cpp_diff.seconds == diff.seconds
    assert cpp_diff.microseconds == diff.microseconds


def test_chrono_duration_subtraction_equivalence():

    date1 = datetime.datetime.today()
    date2 = datetime.datetime.today()

    diff = date2 - date1
    cpp_diff = m.test_chrono4(date2, date1)

    assert cpp_diff.days == diff.days
    assert cpp_diff.seconds == diff.seconds
    assert cpp_diff.microseconds == diff.microseconds


def test_chrono_duration_subtraction_equivalence_date():

    date1 = datetime.date.today()
    date2 = datetime.date.today()

    diff = date2 - date1
    cpp_diff = m.test_chrono4(date2, date1)

    assert cpp_diff.days == diff.days
    assert cpp_diff.seconds == diff.seconds
    assert cpp_diff.microseconds == diff.microseconds


def test_chrono_steady_clock():
    time1 = m.test_chrono5()
    assert isinstance(time1, datetime.timedelta)


def test_chrono_steady_clock_roundtrip():
    time1 = datetime.timedelta(days=10, seconds=10, microseconds=100)
    time2 = m.test_chrono6(time1)

    assert isinstance(time2, datetime.timedelta)

    # They should be identical (no information lost on roundtrip)
    assert time1.days == time2.days
    assert time1.seconds == time2.seconds
    assert time1.microseconds == time2.microseconds


def test_floating_point_duration():
    # Test using a floating point number in seconds
    time = m.test_chrono7(35.525123)

    assert isinstance(time, datetime.timedelta)

    assert time.seconds == 35
    assert 525122 <= time.microseconds <= 525123

    diff = m.test_chrono_float_diff(43.789012, 1.123456)
    assert diff.seconds == 42
    assert 665556 <= diff.microseconds <= 665557


def test_nano_timepoint():
    time = datetime.datetime.now()
    time1 = m.test_nano_timepoint(time, datetime.timedelta(seconds=60))
    assert(time1 == time + datetime.timedelta(seconds=60))
