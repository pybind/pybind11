import pytest

from pybind11_tests import with_ as m


class PythonContextManager:

    def __init__(self, value, swallow_exceptions=False):
        self.value = value
        self.swallow_exceptions = swallow_exceptions
        self.entered = 0
        self.exited = 0
        self.exit_args = None

    def __enter__(self):
        self.entered += 1
        # Return a new instance of a custom class, such that we can check
        # the object's refcount and see if the py::object is moved when possible
        return self.value

    def __exit__(self, *args):
        self.exited += 1
        self.exit_args = args
        return self.swallow_exceptions


class NotAContextManager:
    pass


class AlsoNotAContextManager:

    def __exit__(self, *args):
        return False


class AContextManagerWithAWrongExitSignature:

    def __enter__(self):
        return None

    def __exit__(self):
        return False


class ContextManagerRaisingAtEnter:

    def __init__(self):
        self.entered = 0
        self.exited = 0
        self.exit_args = None

    def __enter__(self):
        self.entered += 1
        raise RuntimeError('This is a test. Please stay calm.')

    def __exit__(self, *args):
        self.exited += 1
        self.exit_args = args
        return False


class ContextManagerRaisingAtExit:

    def __init__(self):
        self.entered = 0
        self.exited = 0
        self.exit_args = None

    def __enter__(self):
        self.entered += 1
        return None

    def __exit__(self, *args):
        self.exited += 1
        self.exit_args = args
        raise RuntimeError('This is a test. Please stay calm.')


@pytest.mark.parametrize("context_manager_cls", [PythonContextManager, m.CppContextManager])
@pytest.mark.parametrize("arg_type", ["no_args", "lvalue_arg", "lvalue_ref_arg",
                                      "lvalue_const_ref_arg", "rvalue_ref_arg"])
def test_arg_types(context_manager_cls, arg_type):
    context_manager = context_manager_cls(arg_type)
    test_func = getattr(m, arg_type)
    expected_value = (None if arg_type == "no_args" else context_manager.value)

    assert test_func(context_manager) == expected_value
    assert context_manager.entered == 1
    assert context_manager.exited == 1
    assert context_manager.exit_args == (None, None, None)


@pytest.mark.parametrize("context_manager_cls", [PythonContextManager, m.CppContextManager])
@pytest.mark.parametrize("exception_type", ["python_exception", "cpp_exception"])
@pytest.mark.parametrize("exception_policy", [m.WithExceptionPolicy.Translate,
                                              m.WithExceptionPolicy.Cascade])
@pytest.mark.parametrize("swallow_exceptions", [True, False])
def test_exceptions(context_manager_cls, exception_type, exception_policy, swallow_exceptions):
    value = '_'.join((exception_type, str(exception_policy), str(swallow_exceptions)))
    context_manager = context_manager_cls(value, swallow_exceptions)
    test_func = getattr(m, exception_type)
    expected_value = value
    should_ignore_exception_at_exit = exception_type == "cpp_exception" and \
        exception_policy == m.WithExceptionPolicy.Cascade
    should_raise = should_ignore_exception_at_exit or not swallow_exceptions

    if should_raise:
        with pytest.raises(RuntimeError, match="test"):
            test_func(context_manager, exception_policy)
    else:
        assert test_func(context_manager, exception_policy) == expected_value
    assert context_manager.entered == 1
    assert context_manager.exited == 1
    if should_ignore_exception_at_exit:
        assert context_manager.exit_args == (None, None, None)
    else:
        assert context_manager.exit_args != (None, None, None)


@pytest.mark.parametrize("context_manager_cls", [PythonContextManager, m.CppContextManager])
@pytest.mark.parametrize("swallow_exceptions", [True, False])
def test_catch_cpp_exception(context_manager_cls, swallow_exceptions):
    context_manager = context_manager_cls("", swallow_exceptions=swallow_exceptions)
    expected_value = "no_exception" if swallow_exceptions else "original_exception"

    assert m.catch_cpp_exception(context_manager) == expected_value
    assert context_manager.entered == 1
    assert context_manager.exited == 1
    assert context_manager.exit_args != (None, None, None)


def test_bad_context_managers():
    with pytest.raises(AttributeError, match="has no attribute '__exit__'"):
        m.lvalue_arg(NotAContextManager())

    with pytest.raises(AttributeError, match="has no attribute '__enter__'"):
        m.lvalue_arg(AlsoNotAContextManager())

    with pytest.raises(TypeError,
                       match="(:?__exit__\\(\\) takes 1 positional argument but 4 were given)|" +
                       "(?:__exit__\\(\\) takes exactly 1 argument \\(4 given\\))"):  # Python 2
        m.lvalue_arg(AContextManagerWithAWrongExitSignature())


def test_raising_context_managers():
    context_manager = ContextManagerRaisingAtEnter()
    with pytest.raises(RuntimeError, match="test"):
        m.lvalue_arg(context_manager)
    assert context_manager.entered == 1
    assert context_manager.exited == 0

    context_manager = ContextManagerRaisingAtExit()
    with pytest.raises(RuntimeError, match="test"):
        m.lvalue_arg(context_manager)
    assert context_manager.entered == 1
    assert context_manager.exited == 1
    assert context_manager.exit_args == (None, None, None)
