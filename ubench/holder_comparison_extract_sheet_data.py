"""Extract mean ratios from holder_comparison.py output."""

# ruff: noqa
# This code has no unit tests.
# ruff cleanup deferred until the next time this code is actually used.

import sys
from typing import List, Optional


def run(args: List[str]) -> None:
    assert len(args) == 1, "log_holder_comparison.txt"

    log_lines = open(args[0]).read().splitlines()

    for ratx in ("_ratS ", "_ratA "):
        print(ratx)
        header = None
        header_row = None
        data_row = None
        data_row_buffer: List[List[str]] = []

        def show() -> Optional[List[str]]:
            if header_row:
                if header is None:  # type: ignore[unreachable]
                    print(",".join(header_row))
                else:
                    assert header == header_row
            if data_row is not None:
                print(",".join(data_row))  # type: ignore[unreachable]
                data_row_buffer.append(data_row)
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
                assert header_row is not None
                assert data_row is not None
                header_row.append("calls")
                data_row.append(flds[0])
                header_row.append("up")
                data_row.append("1.000")
            elif line[2:].startswith(ratx):
                flds = line.split()
                assert len(flds) == 4
                assert header_row is not None
                assert data_row is not None
                header_row.append(line[:2])
                data_row.append(flds[2])
        show()

        assert header_row is not None
        print("Scaled to last column:")
        print(",".join(header_row))
        for data_row in data_row_buffer:
            data_row_rescaled = data_row[:2]
            unit = float(data_row[-1])
            for fld in data_row[2:]:
                data_row_rescaled.append("%.3f" % (float(fld) / unit))
            print(",".join(data_row_rescaled))


if __name__ == "__main__":
    run(args=sys.argv[1:])
