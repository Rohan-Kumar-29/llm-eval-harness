import hashlib
import json
from pathlib import Path
from typing import Optional

import diskcache

_CACHE_DIR = Path(".cache")
_cache: Optional[diskcache.Cache] = None


def _get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        _cache = diskcache.Cache(str(_CACHE_DIR))
    return _cache


def _make_key(model_id: str, prompt_text: str, user_input: str, temperature: float) -> str:
    raw = json.dumps([model_id, prompt_text, user_input, str(temperature)], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def get_cached(model_id: str, prompt_text: str, user_input: str, temperature: float) -> Optional[dict]:
    key = _make_key(model_id, prompt_text, user_input, temperature)
    return _get_cache().get(key)


def set_cached(model_id: str, prompt_text: str, user_input: str, temperature: float, result: dict) -> None:
    key = _make_key(model_id, prompt_text, user_input, temperature)
    _get_cache().set(key, result)
