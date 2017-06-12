#!/bin/bash
# 
# Script to check include/test code for common pybind11 code style errors.
# 
# This script currently checks for
#
# 1. use of tabs instead of spaces
# 2. MSDOS-style CRLF endings
# 3. trailing spaces
# 4. missing space between keyword and parenthesis, e.g.: for(, if(, while(
# 5. Missing space between right parenthesis and brace, e.g. 'for (...){'
# 6. opening brace on its own line. It should always be on the same line as the
#    if/while/for/do statment.
# 
# Invoke as: tools/check-style.sh
#

errors=0
IFS=$'\n'
found=
# The mt=41 sets a red background for matched tabs:
exec 3< <(GREP_COLORS='mt=41' grep $'\t' include/ tests/*.{cpp,py,h} docs/*.rst -rn --color=always)
while read -u 3 f; do
    if [ -z "$found" ]; then
        echo -e '\e[31m\e[01mError: found tabs instead of spaces in the following files:\e[0m'
        found=1
        errors=1
    fi

    echo "    $f"
done

found=
# The mt=41 sets a red background for matched MS-DOS CRLF line endings
exec 3< <(GREP_COLORS='mt=41' grep -IUlr $'\r' include/ tests/*.{cpp,py,h} docs/*.rst --color=always)
while read -u 3 f; do
    if [ -z "$found" ]; then
        echo -e '\e[31m\e[01mError: found CRLF characters in the following files:\e[0m'
        found=1
        errors=1
    fi

    echo "    $f"
done

found=
# The mt=41 sets a red background for matched trailing spaces
exec 3< <(GREP_COLORS='mt=41' grep '\s\+$' include/ tests/*.{cpp,py,h} docs/*.rst -rn --color=always)
while read -u 3 f; do
    if [ -z "$found" ]; then
        echo -e '\e[31m\e[01mError: found trailing spaces in the following files:\e[0m'
        found=1
        errors=1
    fi

    echo "    $f"
done

found=
exec 3< <(grep '\<\(if\|for\|while\|catch\)(\|){' include/ tests/*.{cpp,py,h} -rn --color=always)
while read -u 3 line; do
    if [ -z "$found" ]; then
        echo -e '\e[31m\e[01mError: found the following coding style problems:\e[0m'
        found=1
        errors=1
    fi

    echo "    $line"
done

found=
exec 3< <(GREP_COLORS='mt=41' grep '^\s*{\s*$' include/ docs/*.rst -rn --color=always)
while read -u 3 f; do
    if [ -z "$found" ]; then
        echo -e '\e[31m\e[01mError: braces should occur on the same line as the if/while/.. statement. Found issues in the following files: \e[0m'
        found=1
        errors=1
    fi

    echo "    $f"
done

exit $errors
