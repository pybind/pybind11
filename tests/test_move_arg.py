import pytest
from test_move_arg import *


def test():
    item = Item(42)
    other = item
    access(item)
    print(item)
    del item
    print(other)

def test_consume():
    item = Item(42)
    consume(item)
    access(item)  # should raise, because item is accessed after consumption

def test_consume_twice():
    item = Item(42)
    consume_twice(item, item) # should raise, because item is consumed twice

def test_foo():
    foo()

if __name__ == "__main__":
    test_consume()
