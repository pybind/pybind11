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
# The mt=41 sets a red background for matched tabs:
exec 3< <(GREP_COLORS='mt=41' grep $'\t' include/ tests/ docs/*.rst -rn --color=always)
while read -u 3 f; do
    if [ -z "$found" ]; then
        echo -e '\e[31m\e[01mError: found tabs instead of spaces in the following files:\e[0m'
        found=1
        errors=1
    fi

    echo "    $f"
done

found=
exec 3< <(grep '\<\(if\|for\|while\)(' include/ tests/*.{cpp,py,h} -rn --color=always)
while read -u 3 line; do
    if [ -z "$found" ]; then
        echo -e '\e[31m\e[01mError: found the following coding style problems:\e[0m'
        found=1
        errors=1
    fi

    echo "    $line"
done

exit $errors
