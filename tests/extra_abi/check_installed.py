try:
    import dog

    raise RuntimeError("Broken! Dog must require Pet to be loaded")
except ImportError:
    import pet  # noqa: F401

import dog  # noqa: F811

d = dog.Dog("Bluey")

d.bark()
assert d.name == "Bluey"
