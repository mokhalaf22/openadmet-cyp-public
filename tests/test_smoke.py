"""Smoke tests for the package scaffold.

Real behavioural tests arrive with the modelling code. For now we only assert
the package imports and exposes a version, so `make test`/CI has something to
run against an empty tree.
"""

import cyp


def test_package_imports():
    assert cyp.__version__ is not None
