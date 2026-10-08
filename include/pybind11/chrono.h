/*
    pybind11/chrono.h: Transparent conversion between std::chrono and python's datetime

    Copyright (c) 2016 Trent Houliston <trent@houliston.me> and
                       Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"

#include <chrono>
#include <cmath>
#include <ctime>
#include <mutex>

#if !defined(Py_LIMITED_API)
#    include <datetime.h>
#endif

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
PYBIND11_NAMESPACE_BEGIN(detail)

#if defined(Py_LIMITED_API)
// The datetime C API is not part of the stable ABI: use the Python-level types and attributes.
struct datetime_types {
    PyTypeObject *datetime, *date, *time, *timedelta;
};

inline const datetime_types &get_datetime_types() {
    // The type objects are kept for the lifetime of the process.
    static const datetime_types types = [] {
        module_ m = module_::import("datetime");
        auto get = [&](const char *name) {
            return reinterpret_cast<PyTypeObject *>(object(m.attr(name)).release().ptr());
        };
        return datetime_types{get("datetime"), get("date"), get("time"), get("timedelta")};
    }();
    return types;
}

inline void datetime_import() { (void) get_datetime_types(); }
inline bool is_timedelta(handle h) {
    return PyObject_TypeCheck(h.ptr(), get_datetime_types().timedelta) != 0;
}
inline bool is_datetime(handle h) {
    return PyObject_TypeCheck(h.ptr(), get_datetime_types().datetime) != 0;
}
inline bool is_date(handle h) {
    return PyObject_TypeCheck(h.ptr(), get_datetime_types().date) != 0;
}
inline bool is_time(handle h) {
    return PyObject_TypeCheck(h.ptr(), get_datetime_types().time) != 0;
}
inline int datetime_field(handle h, PyObject *name) {
    auto value = reinterpret_steal<object>(PyObject_GetAttr(h.ptr(), name));
    if (!value) {
        throw error_already_set();
    }
    return value.cast<int>();
}
// One accessor per field: interned attribute lookup here, the C macro otherwise.
#    define PYBIND11_DATETIME_ACCESSOR(fn, attr, c_macro)                                         \
        inline int fn(handle h) {                                                                 \
            static PyObject *name = interned_name(#attr);                                         \
            return datetime_field(h, name);                                                       \
        }

inline PyObject *make_timedelta(int days, int seconds, int microseconds) {
    return PyObject_CallFunction(reinterpret_cast<PyObject *>(get_datetime_types().timedelta),
                                 "iii",
                                 days,
                                 seconds,
                                 microseconds);
}
inline PyObject *
make_datetime(int year, int month, int day, int hour, int minute, int second, int microsecond) {
    return PyObject_CallFunction(reinterpret_cast<PyObject *>(get_datetime_types().datetime),
                                 "iiiiiii",
                                 year,
                                 month,
                                 day,
                                 hour,
                                 minute,
                                 second,
                                 microsecond);
}
#else
inline void datetime_import() {
    if (!PyDateTimeAPI) {
        PyDateTime_IMPORT;
    }
}
inline bool is_timedelta(handle h) { return PyDelta_Check(h.ptr()); }
inline bool is_datetime(handle h) { return PyDateTime_Check(h.ptr()); }
inline bool is_date(handle h) { return PyDate_Check(h.ptr()); }
inline bool is_time(handle h) { return PyTime_Check(h.ptr()); }
#    define PYBIND11_DATETIME_ACCESSOR(fn, attr, c_macro)                                         \
        inline int fn(handle h) { return c_macro(h.ptr()); }
inline PyObject *make_timedelta(int days, int seconds, int microseconds) {
    return PyDelta_FromDSU(days, seconds, microseconds);
}
inline PyObject *
make_datetime(int year, int month, int day, int hour, int minute, int second, int microsecond) {
    return PyDateTime_FromDateAndTime(year, month, day, hour, minute, second, microsecond);
}
#endif

PYBIND11_DATETIME_ACCESSOR(timedelta_days, days, PyDateTime_DELTA_GET_DAYS)
PYBIND11_DATETIME_ACCESSOR(timedelta_seconds, seconds, PyDateTime_DELTA_GET_SECONDS)
PYBIND11_DATETIME_ACCESSOR(timedelta_microseconds, microseconds, PyDateTime_DELTA_GET_MICROSECONDS)
PYBIND11_DATETIME_ACCESSOR(datetime_year, year, PyDateTime_GET_YEAR)
PYBIND11_DATETIME_ACCESSOR(datetime_month, month, PyDateTime_GET_MONTH)
PYBIND11_DATETIME_ACCESSOR(datetime_day, day, PyDateTime_GET_DAY)
PYBIND11_DATETIME_ACCESSOR(datetime_hour, hour, PyDateTime_DATE_GET_HOUR)
PYBIND11_DATETIME_ACCESSOR(datetime_minute, minute, PyDateTime_DATE_GET_MINUTE)
PYBIND11_DATETIME_ACCESSOR(datetime_second, second, PyDateTime_DATE_GET_SECOND)
PYBIND11_DATETIME_ACCESSOR(datetime_microsecond, microsecond, PyDateTime_DATE_GET_MICROSECOND)
PYBIND11_DATETIME_ACCESSOR(time_hour, hour, PyDateTime_TIME_GET_HOUR)
PYBIND11_DATETIME_ACCESSOR(time_minute, minute, PyDateTime_TIME_GET_MINUTE)
PYBIND11_DATETIME_ACCESSOR(time_second, second, PyDateTime_TIME_GET_SECOND)
PYBIND11_DATETIME_ACCESSOR(time_microsecond, microsecond, PyDateTime_TIME_GET_MICROSECOND)
#undef PYBIND11_DATETIME_ACCESSOR

template <typename type>
class duration_caster {
public:
    using rep = typename type::rep;
    using period = typename type::period;

    // signed 25 bits required by the standard.
    using days = std::chrono::duration<int_least32_t, std::ratio<86400>>;

    bool load(handle src, bool) {
        using namespace std::chrono;

        datetime_import();

        if (!src) {
            return false;
        }
        // If invoked with datetime.delta object
        if (is_timedelta(src)) {
            value = type(duration_cast<duration<rep, period>>(
                days(timedelta_days(src)) + seconds(timedelta_seconds(src))
                + microseconds(timedelta_microseconds(src))));
            return true;
        }
        // If invoked with a float we assume it is seconds and convert
        if (PyFloat_Check(src.ptr())) {
            value = type(duration_cast<duration<rep, period>>(
                duration<double>(PyFloat_AsDouble(src.ptr()))));
            return true;
        }
        return false;
    }

    // If this is a duration just return it back
    static const std::chrono::duration<rep, period> &
    get_duration(const std::chrono::duration<rep, period> &src) {
        return src;
    }
    static const std::chrono::duration<rep, period> &
    get_duration(const std::chrono::duration<rep, period> &&) = delete;

    // If this is a time_point get the time_since_epoch
    template <typename Clock>
    static std::chrono::duration<rep, period>
    get_duration(const std::chrono::time_point<Clock, std::chrono::duration<rep, period>> &src) {
        return src.time_since_epoch();
    }

    static handle cast(const type &src, return_value_policy /* policy */, handle /* parent */) {
        using namespace std::chrono;

        // Use overloaded function to get our duration from our source
        // Works out if it is a duration or time_point and get the duration
        auto d = get_duration(src);

        datetime_import();

        // Declare these special duration types so the conversions happen with the correct
        // primitive types (int)
        using dd_t = duration<int, std::ratio<86400>>;
        using ss_t = duration<int, std::ratio<1>>;
        using us_t = duration<int, std::micro>;

        auto dd = duration_cast<dd_t>(d);
        auto subd = d - dd;
        auto ss = duration_cast<ss_t>(subd);
        auto us = duration_cast<us_t>(subd - ss);
        return make_timedelta(dd.count(), ss.count(), us.count());
    }

    PYBIND11_TYPE_CASTER(type, const_name("datetime.timedelta"));
};

inline std::tm *localtime_thread_safe(const std::time_t *time, std::tm *buf) {
#if (defined(__STDC_LIB_EXT1__) && defined(__STDC_WANT_LIB_EXT1__)) || defined(_MSC_VER)
    if (localtime_s(buf, time))
        return nullptr;
    return buf;
#else
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    std::tm *tm_ptr = std::localtime(time);
    if (tm_ptr != nullptr) {
        *buf = *tm_ptr;
    }
    return tm_ptr;
#endif
}

// This is for casting times on the system clock into datetime.datetime instances
template <typename Duration>
class type_caster<std::chrono::time_point<std::chrono::system_clock, Duration>> {
public:
    using type = std::chrono::time_point<std::chrono::system_clock, Duration>;
    bool load(handle src, bool) {
        using namespace std::chrono;

        datetime_import();

        if (!src) {
            return false;
        }

        std::tm cal;
        microseconds msecs;

        if (is_datetime(src)) {
            cal.tm_sec = datetime_second(src);
            cal.tm_min = datetime_minute(src);
            cal.tm_hour = datetime_hour(src);
            cal.tm_mday = datetime_day(src);
            cal.tm_mon = datetime_month(src) - 1;
            cal.tm_year = datetime_year(src) - 1900;
            cal.tm_isdst = -1;
            msecs = microseconds(datetime_microsecond(src));
        } else if (is_date(src)) {
            cal.tm_sec = 0;
            cal.tm_min = 0;
            cal.tm_hour = 0;
            cal.tm_mday = datetime_day(src);
            cal.tm_mon = datetime_month(src) - 1;
            cal.tm_year = datetime_year(src) - 1900;
            cal.tm_isdst = -1;
            msecs = microseconds(0);
        } else if (is_time(src)) {
            cal.tm_sec = time_second(src);
            cal.tm_min = time_minute(src);
            cal.tm_hour = time_hour(src);
            cal.tm_mday = 1;  // This date (day, month, year) = (1, 0, 70)
            cal.tm_mon = 0;   // represents 1-Jan-1970, which is the first
            cal.tm_year = 70; // earliest available date for Python's datetime
            cal.tm_isdst = -1;
            msecs = microseconds(time_microsecond(src));
        } else {
            return false;
        }

        value = time_point_cast<Duration>(system_clock::from_time_t(std::mktime(&cal)) + msecs);
        return true;
    }

    static handle cast(const std::chrono::time_point<std::chrono::system_clock, Duration> &src,
                       return_value_policy /* policy */,
                       handle /* parent */) {
        using namespace std::chrono;

        datetime_import();

        // Get out microseconds, and make sure they are positive, to avoid bug in eastern
        // hemisphere time zones (cfr. https://github.com/pybind/pybind11/issues/2417)
        using us_t = duration<int, std::micro>;
        auto us = duration_cast<us_t>(src.time_since_epoch() % seconds(1));
        if (us.count() < 0) {
            us += duration_cast<us_t>(seconds(1));
        }

        // Subtract microseconds BEFORE `system_clock::to_time_t`, because:
        // > If std::time_t has lower precision, it is implementation-defined whether the value is
        // rounded or truncated. (https://en.cppreference.com/w/cpp/chrono/system_clock/to_time_t)
        std::time_t tt
            = system_clock::to_time_t(time_point_cast<system_clock::duration>(src - us));

        std::tm localtime;
        std::tm *localtime_ptr = localtime_thread_safe(&tt, &localtime);
        if (!localtime_ptr) {
            throw cast_error("Unable to represent system_clock in local time");
        }
        return make_datetime(localtime.tm_year + 1900,
                             localtime.tm_mon + 1,
                             localtime.tm_mday,
                             localtime.tm_hour,
                             localtime.tm_min,
                             localtime.tm_sec,
                             us.count());
    }
    PYBIND11_TYPE_CASTER(type, const_name("datetime.datetime"));
};

// Other clocks that are not the system clock are not measured as datetime.datetime objects
// since they are not measured on calendar time. So instead we just make them timedeltas
// Or if they have passed us a time as a float we convert that
template <typename Clock, typename Duration>
class type_caster<std::chrono::time_point<Clock, Duration>>
    : public duration_caster<std::chrono::time_point<Clock, Duration>> {};

template <typename Rep, typename Period>
class type_caster<std::chrono::duration<Rep, Period>>
    : public duration_caster<std::chrono::duration<Rep, Period>> {};

PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
