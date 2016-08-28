#!/bin/bash
# 
# Script to check include/test code for common pybind11 code style errors.
# Currently just checks for tabs used instead of spaces.
# 
# Invoke as: tools/check-style.sh
#

found=0
for f in `grep $'\t' include/ tests/ docs/*.rst -rl`; do
    if [ "$found" -eq 0 ]; then
        echo -e '\e[31m\e[01mError: found tabs instead of spaces in the following files:\e[0m'
        found=1
    fi

    echo "    $f"
done

exit $found
