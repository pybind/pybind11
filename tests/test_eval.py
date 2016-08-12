

def test_eval(capture):
    from pybind11_tests import example_eval

    with capture:
        example_eval()
    assert capture == """
        eval_statements test
        Hello World!
        eval_statements passed
        eval test
        eval passed
        eval_single_statement test
        eval_single_statement passed
        eval_file test
        eval_file passed
        eval failure test
        eval failure test passed
        eval_file failure test
        eval_file failure test passed
    """
