# -*- coding: utf-8 -*-
"""Simple comparison of holder performances, relative to unique_ptr holder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pybind11_ubench_holder_comparison as m

import collections
import sys
import time

number_bucket_pc = None


def pflush(*args, **kwargs):
    result = print(*args, **kwargs)
    # Using "file" here because it is the name of the built-in keyword argument.
    file = kwargs.get("file", sys.stdout)  # pylint: disable=redefined-builtin
    file.flush()  # file object must have a flush method.
    return result


def run(args):
    if not args:
        size_exponent_min = 0
        size_exponent_max = 16
        size_exponent_step = 4
        call_repetitions_first_pass = 100
        call_repetitions_target_elapsed_secs = 0.1
        num_samples = 10
        selected_holder_type = "all"
    else:
        assert len(args) == 7, (
            "size_exponent_min size_exponent_max size_exponent_step"
            " call_repetitions_first_pass call_repetitions_target_elapsed_secs"
            " num_samples selected_holder_type"
        )
        size_exponent_min = int(args[0])
        size_exponent_max = int(args[1])
        size_exponent_step = int(args[2])
        call_repetitions_first_pass = int(args[3])
        call_repetitions_target_elapsed_secs = float(args[4])
        num_samples = int(args[5])
        selected_holder_type = args[6]
    pflush(
        "command-line arguments:",
        size_exponent_min,
        size_exponent_max,
        size_exponent_step,
        call_repetitions_first_pass,
        "%.3f" % call_repetitions_target_elapsed_secs,
        num_samples,
        selected_holder_type,
    )

    for size_exponent in range(
        size_exponent_min, size_exponent_max + 1, size_exponent_step
    ):
        data_size = 2 ** size_exponent
        pflush(data_size, "data_size")
        ratios = collections.defaultdict(list)
        call_repetitions_dynamic = None
        for _ in range(num_samples):
            row_0 = None
            for nb_label, nb_type in [
                ("up", m.number_bucket_up),
                ("sp", m.number_bucket_sp),
                ("pu", m.number_bucket_pu),
                ("sh", m.number_bucket_sh),
                ("pc", number_bucket_pc),
            ]:
                if nb_label == "pc" and nb_type is None:
                    continue
                if selected_holder_type != "all" and nb_label != selected_holder_type:
                    continue
                nb1 = nb_type(data_size)
                nb2 = nb_type(data_size)
                if call_repetitions_dynamic is None:
                    assert int(round(nb1.sum())) == data_size
                    t0 = time.time()
                    for _ in range(call_repetitions_first_pass):
                        nb1.sum()
                    td_sum = time.time() - t0
                    call_repetitions_dynamic = max(
                        call_repetitions_first_pass,
                        int(
                            call_repetitions_target_elapsed_secs
                            * call_repetitions_first_pass
                            / max(td_sum, 1.0e-6)
                        )
                        + 1,
                    )
                    pflush(call_repetitions_dynamic, "call_repetitions_dynamic")
                assert int(round(nb1.sum())) == data_size
                t0 = time.time()
                for _ in range(call_repetitions_dynamic):
                    nb1.sum()
                td_sum = time.time() - t0
                assert nb1.add(nb2) == data_size
                t0 = time.time()
                for _ in range(call_repetitions_dynamic):
                    nb1.add(nb2)
                td_add = time.time() - t0
                row = [td_sum, td_add]
                if row_0 is None:
                    pflush("     Sum   Add  ratS  ratA")
                    row_0 = row
                else:
                    for curr, prev in zip(row, row_0):
                        if prev:
                            rat = curr / prev
                        else:
                            rat = -1
                        row.append(curr / prev)
                    ratios[nb_label + "_ratS"].append(row[-2])
                    ratios[nb_label + "_ratA"].append(row[-1])
                pflush(nb_label, " ".join(["%.3f" % v for v in row]))
        pflush("          Min  Mean   Max")
        for key, rat in ratios.items():
            print(key, "%5.3f %5.3f %5.3f" % (min(rat), sum(rat) / len(rat), max(rat)))


if __name__ == "__main__":
    run(args=sys.argv[1:])
