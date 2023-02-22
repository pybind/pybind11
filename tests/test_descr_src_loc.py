import pytest

from pybind11_tests import descr_src_loc as m

if m.block_descr_offset is None:
    block_parametrize = (("all_blocks", None),)
else:
    block_parametrize = (
        ("block_descr", (("", 1), ("Abc", 2), ("D", 3), ("Ef", 4))),
        (
            "block_const_name",
            (
                ("G", 1),
                ("Hi", 2),
                ("0", 0),
                ("1", 0),
                ("23", 0),
                ("%", 6),
                ("J", 7),
                ("M", 8),
            ),
        ),
        (
            "block_underscore",
            (
                ("G", 2),
                ("Hi", 3),
                ("0", 0),
                ("1", 0),
                ("23", 0),
                ("%", 7),
                ("J", 8),
                ("M", 9),
            ),
        ),
        ("block_plus", (("NO", 1), ("PQ", 4))),
        ("block_concat", (("R", 1), ("S, T", 2), ("U, V", 6))),
        ("block_type_descr", (("{W}", 1),)),
        ("block_int_to_str", (("", 0), ("4", 0), ("56", 0))),
    )


@pytest.mark.skipif(m.block_descr_offset is None, reason="Not enabled.")
@pytest.mark.parametrize(("block_name", "expected_text_line"), block_parametrize)
def test_block(block_name, expected_text_line):
    offset = getattr(m, f"{block_name}_offset")
    for ix, (expected_text, expected_line) in enumerate(expected_text_line):
        text, file, line = getattr(m, f"{block_name}_c{ix}")
        assert text == expected_text
        if expected_line:
            assert file is not None, expected_text_line
            assert file.endswith("test_descr_src_loc.cpp")
            assert line == offset + expected_line
        else:
            assert file is None
            assert line == 0
