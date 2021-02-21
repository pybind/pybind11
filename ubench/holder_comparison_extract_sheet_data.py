# -*- coding: utf-8 -*-
"""Extract mean ratios from holder_comparison.py output."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys


def run(args):
    assert len(args) == 1, "log_holder_comparison.txt"

    log_lines = open(args[0]).read().splitlines()

    for ratx in ("_ratS ", "_ratA "):
        print(ratx)
        header = None
        header_row = None
        data_row = None

        def show():
            if header_row:
                if header is None:
                    print(",".join(header_row))
                else:
                    assert header == header_row
            if data_row is not None:
                print(",".join(data_row))
            return header_row

        for line in log_lines:
            if line.endswith(" data_size"):
                header = show()
                flds = line.split()
                assert len(flds) == 2
                header_row = ["data_size"]
                data_row = [flds[0]]
            elif line.endswith(" call_repetitions"):
                flds = line.split()
                assert len(flds) == 2
                header_row.append("calls")
                data_row.append(flds[0])
            elif line[2:].startswith(ratx):
                flds = line.split()
                assert len(flds) == 4
                header_row.append(line[:2])
                data_row.append(flds[2])
        show()


if __name__ == "__main__":
    run(args=sys.argv[1:])
