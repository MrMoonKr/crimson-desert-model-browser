"""Cached PAMT parser — pickle cache for fast subsequent loads.

First parse: delegates to paz_parse.parse_pamt (~3-8s for 396K entries)
Subsequent: loads from pickle cache (~0.5-1.5s)
Cache invalidated by PAMT file size + modification time.
"""

import os
import sys
import hashlib
import pickle

# Ensure lazorr410-unpacker is on path
_unpacker_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'lazorr410-unpacker', 'python')
if _unpacker_dir not in sys.path:
    sys.path.insert(0, _unpacker_dir)

from paz_parse import parse_pamt, PazEntry


CACHE_DIR_NAME = ".pamt_cache"


def parse_pamt_cached(pamt_path: str, paz_dir: str = None) -> list[PazEntry]:
    """Parse PAMT with disk cache. Falls back to uncached if cache fails."""
    cache_dir = os.path.join(os.path.dirname(pamt_path), CACHE_DIR_NAME)
    cache_file = _cache_path(pamt_path, cache_dir)

    if cache_file and os.path.isfile(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass

    entries = parse_pamt(pamt_path, paz_dir=paz_dir)

    if cache_file:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(entries, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    return entries


def _cache_path(pamt_path: str, cache_dir: str) -> str | None:
    try:
        stat = os.stat(pamt_path)
        key = f"{os.path.basename(pamt_path)}_{stat.st_size}_{stat.st_mtime_ns}"
        return os.path.join(cache_dir, hashlib.md5(key.encode()).hexdigest() + ".pkl")
    except OSError:
        return None
