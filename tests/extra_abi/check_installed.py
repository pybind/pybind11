from __future__ import annotations

import sys

expect_incompatible = "--expect-incompatible" in sys.argv

# dog alone must fail: it needs pet to register the Pet base class first.
# Otherwise this check would not be exercising cross-extension ABI at all.
try:
    import dog
except Exception:
    pass
else:
    raise SystemExit("Broken: dog imported without pet; the check is not cross-module")

import pet  # noqa: E402

try:
    import dog
except Exception:
    if expect_incompatible:
        print("OK: incompatible internals are isolated, dog did not load")
        sys.exit(0)
    raise

if expect_incompatible:
    raise SystemExit("Broken: dog loaded against pet with mismatched internals")

d = dog.Dog("Bluey")
assert d.bark() == "woof!"
assert d.name == "Bluey"
assert isinstance(d, pet.Pet)
print("OK: cross-version ABI is compatible")
