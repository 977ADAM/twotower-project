from __future__ import annotations


def test_protocols_importable():
    from twotower._src.protocols import _HasConfig, _HasEmbeddings, _HasIDMappings
    assert _HasEmbeddings is not None
    assert _HasIDMappings is not None
    assert _HasConfig is not None
