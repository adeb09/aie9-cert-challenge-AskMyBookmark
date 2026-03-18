"""Disk-based LLM output cache for AskMyBookmark.

Cache files are stored at:
    <cache_dir>/<kind>/<key>.json

Each file is a JSON object:
    {
        "meta": {
            "github_data_hash": "...",
            "prompt_hash": "...",
            "model_hash": "...",
            "kind": "generate_answer",
            "created_at": "2026-03-17T12:34:56"
        },
        "payload": { ... }   # the cached node-output dict
    }

Writes are atomic (write to a `.tmp` sibling then os.replace) so a crash
during a write will never produce a half-written or unreadable cache file.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional


# ---------------------------------------------------------------------------
# Public helper: make_cache_key
# ---------------------------------------------------------------------------

def make_cache_key(*parts: Any) -> str:
    """Return the MD5 hex digest of the concatenated string representations.

    Pass whatever values uniquely identify the LLM call context (query text,
    sorted ratings, prompt hashes, model names, etc.).  The output is used
    as the cache filename.
    """
    raw = "|".join(str(p) for p in parts)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Public helper: md5_of_strings
# ---------------------------------------------------------------------------

def md5_of_strings(*strings: str) -> str:
    """Return the MD5 hex digest of one or more concatenated strings."""
    return hashlib.md5("|".join(strings).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Disk cache class
# ---------------------------------------------------------------------------

class NodeCache:
    """A per-orchestrator-kind, per-user disk cache for LLM node outputs.

    Parameters
    ----------
    cache_dir:
        Root directory for this user's query cache
        (e.g. ``data/cached/<username>/query_cache``).
    github_data_hash:
        Hash of the GitHub index; any change here invalidates all cache
        entries automatically (old files simply won't match on read).
    prompt_hash:
        MD5 hash of all prompt strings relevant to the node kind.
    model_hash:
        MD5 hash of all LLM / embedding model identifiers.
    """

    def __init__(
        self,
        cache_dir: str,
        github_data_hash: str,
        prompt_hash: str,
        model_hash: str,
    ) -> None:
        self._cache_dir       = cache_dir
        self._github_data_hash = github_data_hash
        self._prompt_hash      = prompt_hash
        self._model_hash       = model_hash

    # ------------------------------------------------------------------
    # Low-level read / write
    # ------------------------------------------------------------------

    def _kind_dir(self, kind: str) -> str:
        p = os.path.join(self._cache_dir, kind)
        os.makedirs(p, exist_ok=True)
        return p

    def _path(self, kind: str, key: str) -> str:
        return os.path.join(self._kind_dir(kind), f"{key}.json")

    def get(self, kind: str, key: str) -> Optional[Dict[str, Any]]:
        """Return cached payload dict, or ``None`` on miss / version mismatch."""
        path = self._path(kind, key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            meta = data.get("meta", {})
            # Version check: any of the three hashes must match
            if (
                meta.get("github_data_hash") != self._github_data_hash
                or meta.get("prompt_hash")      != self._prompt_hash
                or meta.get("model_hash")        != self._model_hash
            ):
                return None
            return data.get("payload")
        except Exception:
            return None

    def set(self, kind: str, key: str, payload: Dict[str, Any]) -> None:
        """Atomically write *payload* to the cache."""
        path = self._path(kind, key)
        data = {
            "meta": {
                "github_data_hash": self._github_data_hash,
                "prompt_hash":      self._prompt_hash,
                "model_hash":       self._model_hash,
                "kind":             kind,
                "created_at":       datetime.now(timezone.utc).isoformat(),
            },
            "payload": payload,
        }
        # Atomic write: write to temp file next to the target, then rename
        dir_  = os.path.dirname(path)
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Convenience: get_or_set
    # ------------------------------------------------------------------

    def get_or_set(
        self,
        kind: str,
        key: str,
        compute_fn: Callable[[], Dict[str, Any]],
    ) -> tuple[Dict[str, Any], bool]:
        """Return ``(result, from_cache)`` — calling *compute_fn* on a miss.

        *from_cache* is ``True`` when the value was read from disk, ``False``
        when it was freshly computed and then written to disk.
        """
        cached = self.get(kind, key)
        if cached is not None:
            return cached, True
        result = compute_fn()
        try:
            self.set(kind, key, result)
        except Exception as exc:
            # A write failure is non-fatal — just log and continue
            print(f"[query_cache] WARNING: failed to write {kind}/{key}: {exc}")
        return result, False
