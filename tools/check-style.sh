#!/bin/bash
#
# Script to check include/test code for common pybind11 code style errors.
#
# This script currently checks for
#
# 1. missing space between keyword and parenthesis, e.g.: for(, if(, while(
# 2. Missing space between right parenthesis and brace, e.g. 'for (...){'
# 3. opening brace on its own line. It should always be on the same line as the
#    if/while/for/do statement.
#
# Invoke as: tools/check-style.sh <filenames>
#

check_style_errors=0
IFS=$'\n'


found="$(grep '\<\(if\|for\|while\|catch\)(\|){' "$@" -rn --color=always)"
if [ -n "$found" ]; then
    echo -e '\033[31;01mError: found the following coding style problems:\033[0m'
    check_style_errors=1
    echo "${found//^/    /}"
fi

found="$(awk '
function prefix(filename, lineno) {
    return "    \033[35m" filename "\033[36m:\033[32m" lineno "\033[36m:\033[0m"
}
function mark(pattern, string) { sub(pattern, "\033[01;31m&\033[0m", string); return string }
last && /^\s*{/ {
    print prefix(FILENAME, FNR-1) mark("\\)\\s*$", last)
    print prefix(FILENAME, FNR)   mark("^\\s*{", $0)
    last=""
}
{ last = /(if|for|while|catch|switch)\s*\(.*\)\s*$/ ? $0 : "" }
' "$(find include -type f)" "$@")"
if [ -n "$found" ]; then
    check_style_errors=1
    echo -e '\033[31;01mError: braces should occur on the same line as the if/while/.. statement. Found issues in the following files:\033[0m'
    echo "$found"
fi

exit $check_style_errors
