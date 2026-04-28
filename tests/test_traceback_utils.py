from __future__ import annotations

import pytest

from twotower._src.utils.traceback_utils import filter_traceback


def _raises_from_internal():
    """Simulate an error raised from inside _src (this file is outside _src)."""
    raise ValueError("something went wrong")


def test_filter_traceback_preserves_exception_type_and_message():
    @filter_traceback
    def fn():
        _raises_from_internal()

    with pytest.raises(ValueError, match="something went wrong"):
        fn()


def test_filter_traceback_does_not_suppress_exceptions():
    @filter_traceback
    def fn():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        fn()


def test_filter_traceback_passes_return_value_through():
    @filter_traceback
    def fn(x: int) -> int:
        return x * 2

    assert fn(21) == 42


def test_filter_traceback_passes_args_and_kwargs():
    @filter_traceback
    def fn(a, *, b):
        return a + b

    assert fn(1, b=2) == 3


def test_filter_traceback_preserves_function_name():
    @filter_traceback
    def my_public_method():
        pass

    assert my_public_method.__name__ == "my_public_method"
