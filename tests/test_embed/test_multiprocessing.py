from __future__ import annotations

import multiprocessing
import multiprocessing.connection


def f(tx: multiprocessing.connection.Connection, n: int):
    tx.send(n**2)
    tx.close()


def main():
    multiprocessing.set_start_method("spawn")
    assert multiprocessing.get_start_method() == "spawn", "expected spawn"

    rx, tx = multiprocessing.Pipe()
    proc = multiprocessing.Process(target=f, args=(tx, 5))
    proc.start()

    value: int | None = None
    for _ in range(5):
        if rx.poll(1.0):
            value = rx.recv()
            break
    rx.close()
    proc.join(1.0)

    assert value is not None, "no data received"
    assert value == 5**2, f"expected {5**2} got {value}"
