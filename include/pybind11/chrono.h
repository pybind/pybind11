/*
    pybind11/chrono.h: Transparent conversion between std::chrono and python's datetime

    Copyright (c) 2016 Trent Houliston <trent@houliston.me> and
                       Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "pybind11.h"
#include <cmath>
#include <chrono>
#include <datetime.h>

NAMESPACE_BEGIN(pybind11)
NAMESPACE_BEGIN(detail)

template <typename Rep, typename Period> class type_caster<std::chrono::duration<Rep, Period>> {
public:
    typedef std::chrono::duration<Rep, Period> type;
    typedef std::chrono::duration<std::chrono::hours::rep, std::ratio<86400>> days;

    bool load(handle src, bool) {
        using namespace std::chrono;

        // Lazy initialise the PyDateTime import
        if (!PyDateTimeAPI) { PyDateTime_IMPORT; }

        if (!src) return false;
        // If they have passed us a datetime.delta object
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
        // If they have passed us a float we can assume it is seconds and convert
        else if (PyFloat_Check(src.ptr())) {
            double val = PyFloat_AsDouble(src.ptr());
            // Multiply by the reciprocal of the ratio and round
            value = type(std::lround(val * type::period::den / type::period::num));
            return true;
        }
        else return false;
    }

    static handle cast(const std::chrono::duration<Rep, Period> &src, return_value_policy /* policy */, handle /* parent */) {
        using namespace std::chrono;
        if (!PyDateTimeAPI) { PyDateTime_IMPORT; }

        // Declare these special duration types so the conversions happen with the correct primitive types (int)
        using dd_t = duration<int, std::ratio<86400>>;
        using ss_t = duration<int, std::ratio<1>>;
        using us_t = duration<int, std::micro>;

        return PyDelta_FromDSU(
              duration_cast<dd_t>(src).count()
            , duration_cast<ss_t>(src % days(1)).count()
            , duration_cast<us_t>(src % seconds(1)).count());
    }
    PYBIND11_TYPE_CASTER(type, _("datetime.timedelta"));
};

// This is for casting times on the system clock into datetime.datetime instances
template <typename Duration> class type_caster<std::chrono::time_point<std::chrono::system_clock, Duration>> {
public:
    typedef std::chrono::time_point<std::chrono::system_clock, Duration> type;
    bool load(handle src, bool) {
        using namespace std::chrono;

        // Lazy initialise the PyDateTime import
        if (!PyDateTimeAPI) { PyDateTime_IMPORT; }

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

        // Lazy initialise the PyDateTime import
        if (!PyDateTimeAPI) { PyDateTime_IMPORT; }

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

// Other clocks that are not the system clock are not measured as datetime.datetime objects
// since they are not measured on calendar time. So instead we just make them timedeltas
// Or if they have passed us a time as a float we convert that
template <typename Clock, typename Duration> class type_caster<std::chrono::time_point<Clock, Duration>> {
public:
    typedef std::chrono::time_point<Clock, Duration> type;
    typedef std::chrono::duration<std::chrono::hours::rep, std::ratio<86400>> days;

    bool load(handle src, bool) {
        using namespace std::chrono;
        if (!PyDateTimeAPI) { PyDateTime_IMPORT; }

        // If they have passed us a datetime.delta object
        if (PyDelta_Check(src.ptr())) {
            // The accessor macros for timedelta exist in some versions of python but not others (e.g. Mac OSX default python)
            // Therefore we are just doing what the macros do explicitly
            const PyDateTime_Delta* delta = reinterpret_cast<PyDateTime_Delta*>(src.ptr());
            value = time_point<Clock, Duration>(
                  days(delta->days)
                + seconds(delta->seconds)
                + microseconds(delta->microseconds));
            return true;
        }
        // If they have passed us a float we can assume it is seconds and convert
        else if (PyFloat_Check(src.ptr())) {
            double val = PyFloat_AsDouble(src.ptr());
            value = time_point<Clock, Duration>(Duration(std::lround((val / Clock::period::num) * Clock::period::den)));
            return true;
        }
        else return false;
    }

    static handle cast(const std::chrono::time_point<Clock, Duration> &src, return_value_policy /* policy */, handle /* parent */) {
        using namespace std::chrono;

        // Lazy initialise the PyDateTime import
        if (!PyDateTimeAPI) { PyDateTime_IMPORT; }

        // Declare these special duration types so the conversions happen with the correct primitive types (int)
        using dd_t = duration<int, std::ratio<86400>>;
        using ss_t = duration<int, std::ratio<1>>;
        using us_t = duration<int, std::micro>;

        Duration d = src.time_since_epoch();

        return PyDelta_FromDSU(
              duration_cast<dd_t>(d).count()
            , duration_cast<ss_t>(d % days(1)).count()
            , duration_cast<us_t>(d % seconds(1)).count());
    }
    PYBIND11_TYPE_CASTER(type, _("datetime.timedelta"));
};

NAMESPACE_END(detail)
NAMESPACE_END(pybind11)
