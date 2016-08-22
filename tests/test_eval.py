import os


def test_evals(capture):
    from pybind11_tests import (test_eval_statements, test_eval, test_eval_single_statement,
                                test_eval_file, test_eval_failure, test_eval_file_failure)

    with capture:
        assert test_eval_statements()
    assert capture == "Hello World!"

    assert test_eval()
    assert test_eval_single_statement()

    filename = os.path.join(os.path.dirname(__file__), "test_eval_call.py")
    assert test_eval_file(filename)

    assert test_eval_failure()
    assert test_eval_file_failure()
