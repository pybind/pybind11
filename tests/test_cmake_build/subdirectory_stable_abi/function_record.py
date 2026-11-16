from __future__ import annotations

import function_record_full as full
import function_record_stable as stable
import function_record_stable_peer as peer

for consumer, producer in (
    (stable, full),
    (full, stable),
    (stable, peer),
    (peer, stable),
):
    # Incompatible wrappers must use the Python callback path. Compatible foreign
    # records should still allow extracting the original C++ function pointer.
    compatible = consumer.opaque_function_record == producer.opaque_function_record
    result = consumer.call_callback(producer.add_one)
    assert result == (42, compatible), (
        consumer.__name__,
        producer.__name__,
        result,
    )

print("Function-record compatibility and cross-module callbacks passed")
