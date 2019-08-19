import asyncio
import pytest
from pybind11_tests import async_module as m


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


async def get_await_result(x):
    return await x


def test_await(event_loop):
    assert 5 == event_loop.run_until_complete(get_await_result(m.SupportsAsync()))


def test_await_missing(event_loop):
    with pytest.raises(TypeError):
        event_loop.run_until_complete(get_await_result(m.DoesNotSupportAsync()))
