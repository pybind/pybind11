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

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

template <typename Rep, typename Period> class type_caster<std::chrono::duration<Rep, Period>> {
public:
    typedef std::chrono::duration<Rep, Period> type;
    typedef std::chrono::duration<std::chrono::hours::rep, std::ratio<86400>> days;

    bool load(handle src, bool) {
        using namespace std::chrono;
        PyDateTime_IMPORT;

        if (!src) return false;
        if (PyDelta_Check(src.ptr())) {
            // The accessor macros for timedelta exist in some versions of python but not others (e.g. Mac OSX default python)
            // Therefore we are just doing what the macros do explicitly
            const PyDateTime_Delta* delta = reinterpret_cast<PyDateTime_Delta*>(src.ptr());
            value = duration_cast<duration<Rep, Period>>(
                  days(delta->days)
                + seconds(delta->seconds)
                + microseconds(delta->microseconds));
            return true;
        }
        else return false;
    }

    static handle cast(const std::chrono::duration<Rep, Period> &src, return_value_policy /* policy */, handle /* parent */) {
        using namespace std::chrono;
        PyDateTime_IMPORT;

        // Declare these special duration types so the conversions happen with the correct primitive types (int)
        typedef duration<int, std::ratio<86400>> dd_t;
        typedef duration<int, std::ratio<1>> ss_t;
        typedef duration<int, std::micro> us_t;

        return PyDelta_FromDSU(
              duration_cast<dd_t>(src).count()
            , duration_cast<ss_t>(src % days(1)).count()
            , duration_cast<us_t>(src % seconds(1)).count());
    }
    PYBIND11_TYPE_CASTER(type, _("datetime.timedelta"));
};

template <typename Duration> class type_caster<std::chrono::time_point<std::chrono::system_clock, Duration>> {
public:
    typedef std::chrono::time_point<std::chrono::system_clock, Duration> type;
    bool load(handle src, bool) {
        using namespace std::chrono;
        PyDateTime_IMPORT;

        if (!src) return false;
        if (PyDateTime_Check(src.ptr())) {
            std::tm cal;
            cal.tm_sec   = PyDateTime_DATE_GET_SECOND(src.ptr());
            cal.tm_min   = PyDateTime_DATE_GET_MINUTE(src.ptr());
            cal.tm_hour  = PyDateTime_DATE_GET_HOUR(src.ptr());
            cal.tm_mday  = PyDateTime_GET_DAY(src.ptr());
            cal.tm_mon   = PyDateTime_GET_MONTH(src.ptr()) - 1;
            cal.tm_year  = PyDateTime_GET_YEAR(src.ptr()) - 1900;
            cal.tm_isdst = -1;

            value = system_clock::from_time_t(mktime(&cal)) + microseconds(PyDateTime_DATE_GET_MICROSECOND(src.ptr()));
            return true;
        }
        else return false;
    }

    static handle cast(const std::chrono::time_point<std::chrono::system_clock, Duration> &src, return_value_policy /* policy */, handle /* parent */) {
        using namespace std::chrono;
        PyDateTime_IMPORT;

        time_t tt = system_clock::to_time_t(src);
        // this function uses static memory so it's best to copy it out asap just in case
        tm *ltime = localtime(&tt);
        tm localtime = *ltime;

        // Declare these special duration types so the conversions happen with the correct primitive types (int)
        using us_t = duration<int, std::micro>;

        return PyDateTime_FromDateAndTime(localtime.tm_year + 1900
                                          , localtime.tm_mon + 1
                                          , localtime.tm_mday
                                          , localtime.tm_hour
                                          , localtime.tm_min
                                          , localtime.tm_sec
                                          , (duration_cast<us_t>(src.time_since_epoch() % seconds(1))).count());
    }
    PYBIND11_TYPE_CASTER(type, _("datetime.datetime"));
};

template <typename Clock, typename Duration> class type_caster<std::chrono::time_point<Clock, Duration>> {
public:
    typedef std::chrono::time_point<Clock, Duration> type;
    bool load(handle src, bool) {
        using namespace std::chrono;
        PyDateTime_IMPORT;

        if (!src) return false;
        if (PyTime_Check(src.ptr())) {
            value = type(duration_cast<Duration>(
                   hours(PyDateTime_TIME_GET_HOUR(src.ptr()))
                 + minutes(PyDateTime_TIME_GET_MINUTE(src.ptr()))
                 + seconds(PyDateTime_TIME_GET_SECOND(src.ptr()))
                 + microseconds(PyDateTime_TIME_GET_MICROSECOND(src.ptr()))
            ));
            return true;
        }
        else return false;
    }

    static handle cast(const std::chrono::time_point<Clock, Duration> &src, return_value_policy /* policy */, handle /* parent */) {
        using namespace std::chrono;
        PyDateTime_IMPORT;

        // Declare these special duration types so the conversions happen with the correct primitive types (int)
        typedef duration<int, std::ratio<3600>> hh_t;
        typedef duration<int, std::ratio<60>> mm_t;
        typedef duration<int, std::ratio<1>> ss_t;
        typedef duration<int, std::micro> us_t;

        Duration d = src.time_since_epoch();
        return PyTime_FromTime(duration_cast<hh_t>(d).count()
                               , duration_cast<mm_t>(d % hours(1)).count()
                               , duration_cast<ss_t>(d % minutes(1)).count()
                               , duration_cast<us_t>(d % seconds(1)).count());
    }
    PYBIND11_TYPE_CASTER(type, _("datetime.time"));
};

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)
