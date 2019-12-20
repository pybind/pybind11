import pytest
from test_move_arg import Item, access


def test():
    item = Item(42)
    other = item
    access(item)
    print(item)
    del item
    print(other)

if __name__ == "__main__":
    test()
