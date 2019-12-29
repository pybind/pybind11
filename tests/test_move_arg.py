import pytest
from test_move_arg import *


def test():
    item = Item(42)
    other = item
    access(item)
    print(item)
    del item
    print(other)

def test_produce():
    item = Item(42)
    access(item)
    consume(item)
    access(item)

def test_foo():
    foo()

if __name__ == "__main__":
    test_produce()
