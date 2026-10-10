from __future__ import annotations

import datetime
import time

import pytest

import env  # noqa: F401
from pybind11_tests import chrono as m


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
    assert diff == datetime.timedelta(0)


@pytest.mark.parametrize("day", [1, 2])
def test_chrono_system_clock_roundtrip_epoch(day):
    # Python's fold-aware naive timestamp implementation probes the previous
    # day, which Windows cannot represent before the epoch.
    value = datetime.datetime(1970, 1, day, 12)
    assert m.test_chrono2(value) == value


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
    assert date1 == date2

    # There should be no time information
    assert time2.hour == 0
    assert time2.minute == 0
    assert time2.second == 0
    assert time2.microsecond == 0


@pytest.fixture
def local_timezone(monkeypatch):
    if not hasattr(time, "tzset"):
        pytest.skip("Changing the local timezone requires time.tzset")
    try:
        with monkeypatch.context() as patch:

            def set_timezone(tz):
                patch.setenv("TZ", tz)
                time.tzset()

            yield set_timezone
    finally:
        time.tzset()


def epoch_us(utc):
    delta = utc - datetime.datetime(1970, 1, 1)
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


# POSIX rules make the core cases independent of the host's IANA timezone data.
DST_CASES = [
    pytest.param(
        "PST8PDT,M3.2.0/2,M11.1.0/2",
        datetime.datetime(2026, 11, 1, 1, 30, 42, 123456, fold=0),
        datetime.timedelta(hours=-7),
        id="fall-first",
    ),
    pytest.param(
        "PST8PDT,M3.2.0/2,M11.1.0/2",
        datetime.datetime(2026, 11, 1, 1, 30, 42, 123456, fold=1),
        datetime.timedelta(hours=-8),
        id="fall-second",
    ),
    pytest.param(
        "PST8PDT,M3.2.0/2,M11.1.0/2",
        datetime.datetime(2026, 3, 8, 2, 30, 42, 654321, fold=0),
        datetime.timedelta(hours=-8),
        id="spring-gap-first",
    ),
    pytest.param(
        "PST8PDT,M3.2.0/2,M11.1.0/2",
        datetime.datetime(2026, 3, 8, 2, 30, 42, 654321, fold=1),
        datetime.timedelta(hours=-7),
        id="spring-gap-second",
    ),
    pytest.param(
        "XST-10XDT-10:30,M3.2.0/2,M11.1.0/2",
        datetime.datetime(2026, 11, 1, 1, 45, 42, 123456, fold=0),
        datetime.timedelta(hours=10, minutes=30),
        id="half-hour-first",
    ),
    pytest.param(
        "XST-10XDT-10:30,M3.2.0/2,M11.1.0/2",
        datetime.datetime(2026, 11, 1, 1, 45, 42, 123456, fold=1),
        datetime.timedelta(hours=10),
        id="half-hour-second",
    ),
]


@pytest.mark.parametrize(("tz", "local", "offset"), DST_CASES)
def test_chrono_system_clock_load_dst(tz, local, offset, local_timezone):
    local_timezone(tz)
    assert m.test_chrono_system_clock_as_us(local) == epoch_us(local - offset)


@pytest.mark.parametrize(("tz", "local", "offset"), DST_CASES)
def test_chrono_system_clock_cast_dst(tz, local, offset, local_timezone):
    local_timezone(tz)
    value = epoch_us(local - offset)
    seconds, microseconds = divmod(value, 1_000_000)
    expected = datetime.datetime.fromtimestamp(seconds).replace(
        microsecond=microseconds
    )
    result = m.test_chrono_system_clock_from_us(value)
    assert result == expected
    # datetime equality ignores fold when comparing naive values.
    assert result.fold == expected.fold
    assert result.tzinfo is None


@pytest.mark.parametrize("fold", [0, 1])
def test_chrono_system_clock_non_dst_fold(fold, local_timezone):
    local_timezone("Europe/Kyiv")
    first = epoch_us(datetime.datetime(1990, 6, 30, 21, 30))
    second = first + 3600_000_000
    # Both sides of this political offset change are daylight time. Skip if
    # the system lacks the corresponding IANA timezone data.
    for value in (first, second):
        parts = time.localtime(value // 1_000_000)
        if parts[:6] != (1990, 7, 1, 1, 30, 0) or parts.tm_isdst != 1:
            pytest.skip("Europe/Kyiv's 1990 offset change is unavailable")
    local = datetime.datetime(1990, 7, 1, 1, 30, fold=fold)
    value = first if fold == 0 else second
    assert m.test_chrono_system_clock_as_us(local) == value
    result = m.test_chrono_system_clock_from_us(value)
    assert result == local
    assert result.fold == fold


@pytest.mark.parametrize(
    "value",
    [
        -1_000_001,
        -1,
        0,
        1,
        1_000_001,
        epoch_us(datetime.datetime(1900, 1, 1, 0, 0, 0, 123456)),
        epoch_us(datetime.datetime(2200, 1, 1)),
        epoch_us(datetime.datetime(2250, 1, 1, 0, 0, 0, 123456)),
    ],
)
def test_chrono_system_clock_microseconds(value, local_timezone):
    local_timezone("UTC0")
    try:
        datetime.datetime.fromtimestamp(value // 1_000_000)
    except (OverflowError, OSError):
        pytest.skip("Timestamp is outside the platform's supported range")
    expected = datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=value)
    assert m.test_chrono_system_clock_as_us(expected) == value
    assert m.test_chrono_system_clock_from_us(value) == expected


class DatetimeWithOverriddenTimestamp(datetime.datetime):
    def timestamp(self):
        raise AssertionError("The system-clock caster should read datetime fields")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (datetime.date(2026, 11, 1), datetime.datetime(2026, 11, 1)),
        (
            datetime.time(12, 34, 56, 123456),
            datetime.datetime(1970, 1, 1, 12, 34, 56, 123456),
        ),
        (
            datetime.time(
                12,
                34,
                56,
                123456,
                tzinfo=datetime.timezone(datetime.timedelta(hours=12)),
                fold=1,
            ),
            datetime.datetime(1970, 1, 1, 12, 34, 56, 123456),
        ),
        (
            datetime.datetime(
                2026,
                11,
                1,
                1,
                30,
                42,
                123456,
                tzinfo=datetime.timezone(datetime.timedelta(hours=12)),
            ),
            datetime.datetime(2026, 11, 1, 1, 30, 42, 123456),
        ),
        (
            DatetimeWithOverriddenTimestamp(2026, 11, 1, 1, 30, 42, 123456),
            datetime.datetime(2026, 11, 1, 1, 30, 42, 123456),
        ),
    ],
)
def test_chrono_system_clock_input_fields(source, expected, local_timezone):
    local_timezone("UTC0")
    assert m.test_chrono_system_clock_as_us(source) == epoch_us(expected)


SKIP_TZ_ENV_ON_WIN = pytest.mark.skipif(
    "env.WIN", reason="TZ environment variable only supported on POSIX"
)


@pytest.mark.parametrize(
    "time1",
    [
        datetime.datetime.today().time(),
        datetime.time(0, 0, 0),
        datetime.time(0, 0, 0, 1),
        datetime.time(0, 28, 45, 109827),
        datetime.time(0, 59, 59, 999999),
        datetime.time(1, 0, 0),
        datetime.time(5, 59, 59, 0),
        datetime.time(5, 59, 59, 1),
    ],
)
@pytest.mark.parametrize(
    "tz",
    [
        None,
        pytest.param("Europe/Brussels", marks=SKIP_TZ_ENV_ON_WIN),
        pytest.param("Asia/Pyongyang", marks=SKIP_TZ_ENV_ON_WIN),
        pytest.param("America/New_York", marks=SKIP_TZ_ENV_ON_WIN),
    ],
)
def test_chrono_system_clock_roundtrip_time(time1, tz, request):
    if tz is not None:
        request.getfixturevalue("local_timezone")(f"/usr/share/zoneinfo/{tz}")

    # Roundtrip the time
    datetime2 = m.test_chrono2(time1)
    date2 = datetime2.date()
    time2 = datetime2.time()

    # The returned value should be a datetime
    assert isinstance(datetime2, datetime.datetime)
    assert isinstance(date2, datetime.date)
    assert isinstance(time2, datetime.time)

    # Hour, Minute, Second & Microsecond should be the same after the round trip
    assert time1 == time2

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

    assert cpp_diff == diff

    # Negative timedelta roundtrip
    diff = datetime.timedelta(microseconds=-1)
    cpp_diff = m.test_chrono3(diff)

    assert cpp_diff == diff


def test_chrono_duration_subtraction_equivalence():
    date1 = datetime.datetime.today()
    date2 = datetime.datetime.today()

    diff = date2 - date1
    cpp_diff = m.test_chrono4(date2, date1)

    assert cpp_diff == diff


def test_chrono_duration_subtraction_equivalence_date():
    date1 = datetime.date.today()
    date2 = datetime.date.today()

    diff = date2 - date1
    cpp_diff = m.test_chrono4(date2, date1)

    assert cpp_diff == diff


def test_chrono_steady_clock():
    time1 = m.test_chrono5()
    assert isinstance(time1, datetime.timedelta)


def test_chrono_steady_clock_roundtrip():
    time1 = datetime.timedelta(days=10, seconds=10, microseconds=100)
    time2 = m.test_chrono6(time1)

    assert isinstance(time2, datetime.timedelta)

    # They should be identical (no information lost on roundtrip)
    assert time1 == time2


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
    assert time1 == time + datetime.timedelta(seconds=60)


def test_chrono_different_resolutions():
    resolutions = m.different_resolutions()
    time = datetime.datetime.now()
    resolutions.timestamp_h = time
    resolutions.timestamp_m = time
    resolutions.timestamp_s = time
    resolutions.timestamp_ms = time
    resolutions.timestamp_us = time
