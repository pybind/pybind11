#!/bin/bash
# 
# Script to check include/test code for common pybind11 code style errors.
# Currently just checks for tabs used instead of spaces.
# 
# Invoke as: tools/check-style.sh
#

errors=0
IFS=$'\n'
found=
grep $'\t' include/ tests/ docs/*.rst -rl | while read f; do
    if [ -z "$found" ]; then
        echo -e '\e[31m\e[01mError: found tabs instead of spaces in the following files:\e[0m'
        found=1
        errors=1
    fi

    echo "    $f"
done

found=
grep '\<\(if\|for\|while\)(' include/ tests/* -r --color=always | while read line; do
    if [ -z "$found" ]; then
        echo -e '\e[31m\e[01mError: found the following coding style problems:\e[0m'
        found=1
        errors=1
    fi

    echo "    $line"
done

exit $errors
