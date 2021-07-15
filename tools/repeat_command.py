# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import sys


def run(args):
    num_repeats = int(args[0])
    cmd_and_args = args[1:]
    assert num_repeats > 0
    assert cmd_and_args
    print("REPEAT_COMMAND:CMD_AND_ARGS", cmd_and_args)
    print()
    sys.stdout.flush()
    first_non_zero_retcode = 0
    for ix in range(num_repeats):
        print("REPEAT_COMMAND:CALL", ix + 1)
        sys.stdout.flush()
        retcode = subprocess.call(cmd_and_args)
        print("REPEAT_COMMAND:RETCODE", retcode)
        print()
        sys.stdout.flush()
        if retcode and not first_non_zero_retcode:
            first_non_zero_retcode = retcode
    print("REPEAT_COMMAND:FIRST_NON_ZERO_RETCODE", first_non_zero_retcode)
    print()
    sys.stdout.flush()
    return first_non_zero_retcode


if __name__ == "__main__":
    sys.exit(run(args=sys.argv[1:]))
