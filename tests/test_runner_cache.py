import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from evalharness.cache import _make_key, get_cached, set_cached
from evalharness.clients import GenerationResult


def test_cache_key_is_deterministic():
    k1 = _make_key("gemini/gemini-2.5-flash", "extract this", "some input", 0.0)
    k2 = _make_key("gemini/gemini-2.5-flash", "extract this", "some input", 0.0)
    assert k1 == k2


def test_cache_key_differs_on_model():
    k1 = _make_key("gemini/gemini-2.5-flash", "prompt", "input", 0.0)
    k2 = _make_key("groq/llama-3.3-70b-versatile", "prompt", "input", 0.0)
    assert k1 != k2


def test_cache_key_differs_on_temperature():
    k1 = _make_key("model", "prompt", "input", 0.0)
    k2 = _make_key("model", "prompt", "input", 0.5)
    assert k1 != k2


def test_cache_roundtrip(tmp_path, monkeypatch):
    """set_cached then get_cached returns the same dict."""
    import diskcache
    cache = diskcache.Cache(str(tmp_path / "testcache"))
    monkeypatch.setattr("evalharness.cache._cache", cache)

    payload = {"text": "hello", "latency_ms": 100.0, "error": None}
    set_cached("model", "prompt", "input", 0.0, payload)
    result = get_cached("model", "prompt", "input", 0.0)
    assert result == payload


def test_cache_miss_returns_none(tmp_path, monkeypatch):
    import diskcache
    cache = diskcache.Cache(str(tmp_path / "testcache2"))
    monkeypatch.setattr("evalharness.cache._cache", cache)

    result = get_cached("model", "prompt", "input_never_seen", 0.0)
    assert result is None
