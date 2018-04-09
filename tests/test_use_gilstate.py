import pytest

pytestmark = pytest.requires_threading

with pytest.suppress(ImportError):
    import threading
    from pybind11_tests import use_gilstate as ug


def run_in_thread(target, args=()):
    def thread_routine(target, args):
        threading.current_thread()._return = target(*args)

    thread = threading.Thread(target=thread_routine, args=(target, args))
    thread.start()
    thread.join()
    return thread._return


def test_use_gilstate():
    assert run_in_thread(target=ug.check_use_gilstate, args=(False,))
    assert run_in_thread(target=ug.check_use_gilstate, args=(True,))


def test_default():
    assert run_in_thread(target=ug.check_default)


def test_cthread():
    assert ug.check_use_gilstate_cthread(False)
    assert ug.check_use_gilstate_cthread(True)
