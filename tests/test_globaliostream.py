from pybind11_tests import globaliostream as m


def test_captured(capture):
    with capture:
        m.c_output(" <OK > ")
    assert capture == ""

    pyob = m.redirect_output()

    with capture:
        m.c_output("I've been redirected to Python, I hope!")
    assert capture == "I've been redirected to Python, I hope!"

    pyob  # to silence warning about unused variable


def test_delete(capture):
    # Deleting pyob should remove io modifier, but might not delete where expected in older Pythons
    # So just verifying that the capture doesn't "leak" (pretty unlikely in pytest)

    with capture:
        m.c_output(" <OK > ")

    assert capture == ""
