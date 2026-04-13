from __future__ import annotations

import importlib.metadata

import softisoics as m


def test_version() -> None:
    assert importlib.metadata.version("softisoics") == m.__version__
