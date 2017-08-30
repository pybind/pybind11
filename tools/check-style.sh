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

check_style_errors=0
IFS=$'\n'

found="$( GREP_COLORS='mt=41' GREP_COLOR='41' grep $'\t' include tests/*.{cpp,py,h} docs/*.rst -rn --color=always )"
if [ ! -z "$found" ]; then
    # The mt=41 sets a red background for matched tabs:
    echo -e '\033[31m\033[01mError: found CRLF characters in the following files:\033[0m'
    check_style_errors=1
    echo "$found" | while read line; do
        echo -e '\033[31m\033[01mError: found tabs instead of spaces in the following files:\033[0m'
    done
fi


found="$( grep -IUlr $'\r' include tests/*.{cpp,py,h} docs/*.rst --color=always )"
if [ ! -z "$found" ]; then
    echo -e '\033[31m\033[01mError: found CRLF characters in the following files:\033[0m'
    check_style_errors=1
    echo "$found" | while read line; do
        echo "    $line"
    done
fi

found="$(GREP_COLORS='mt=41' GREP_COLOR='41' grep '[[:blank:]]\+$' include tests/*.{cpp,py,h} docs/*.rst -rn --color=always )"
if [ ! -z "$found" ]; then
    # The mt=41 sets a red background for matched trailing spaces
    echo -e '\033[31m\033[01mError: found trailing spaces in the following files:\033[0m'
    check_style_errors=1
    echo "$found" | while read line; do
        echo "    $line"
    done
fi

found="$(grep '\<\(if\|for\|while\|catch\)(\|){' include tests/*.{cpp,py,h} -rn --color=always)"
if [ ! -z "$found" ]; then
    echo -e '\033[31m\033[01mError: found the following coding style problems:\033[0m'
    check_style_errors=1

    echo "$found" | while read line; do
        echo "    $line"
    done
fi

found="$(GREP_COLORS='mt=41' GREP_COLOR='41' grep '^\s*{\s*$' include docs/*.rst -rn --color=always)"
if [ ! -z "$found" ] ; then
    check_style_errors=1
    echo -e '\033[31m\033[01mError: braces should occur on the same line as the if/while/.. statement. Found issues in the following files: \033[0m'
    echo "$found" | while read line; do
        echo "    $line"
    done
fi

exit $check_style_errors
