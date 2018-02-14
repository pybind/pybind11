import pytest

pytestmark = pytest.requires_threading

with pytest.suppress(ImportError):
    import threading
    from pybind11_tests import threads as t


def test_threads():
    def pythread_routine():
        threading.current_thread()._return = t.check_pythread()

    thread = threading.Thread(target=pythread_routine)
    thread.start()
    thread.join()
    assert thread._return

    assert t.check_cthread()
