"""Resume cache with TTL and optional disk persistence."""

from __future__ import annotations

import asyncio
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


def _utcnow() -> datetime:
    # Returns aware datetime (with tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


@dataclass
class ResumeCacheEntry:
    key: str
    resume_text: str
    summary: str
    match_rate: float
    created_at: datetime 
    expires_at: datetime
    metadata: dict[str, Any]

    def is_expired(self, now: datetime | None = None) -> bool:
        return (now or _utcnow()) >= self.expires_at


class ResumeCache:
    """Simple LRU + TTL cache for generated resumes with optional disk persistence."""

    def __init__(
        self,
        max_entries: int = 200,
        ttl_seconds: int = 86400,
        cache_path: str | None = None,
    ) -> None:
        self._max_entries = max_entries
        self._ttl = timedelta(seconds=ttl_seconds)
        self._lock = asyncio.Lock()
        self._entries: OrderedDict[str, ResumeCacheEntry] = OrderedDict()
        self._cache_path = Path(cache_path) if cache_path else None
        if self._cache_path:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def build_key(self, job_description: str, profile_fingerprint: str) -> str:
        payload = f"{job_description.strip()}|{profile_fingerprint}"
        return sha256(payload.encode("utf-8")).hexdigest()

    async def get(self, key: str) -> ResumeCacheEntry | None:
        async with self._lock:
            entry = self._entries.get(key)
            if not entry:
                return None
            if entry.is_expired():
                self._entries.pop(key, None)
                self._persist_to_disk()
                return None
            self._entries.move_to_end(key)
            return entry

    async def set(self, entry: ResumeCacheEntry) -> None:
        async with self._lock:
            if entry.is_expired():
                return
            self._entries[entry.key] = entry
            self._entries.move_to_end(entry.key)
            self._evict_if_needed()
            self._persist_to_disk()

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def _load_from_disk(self) -> None:
        if not self._cache_path or not self._cache_path.exists():
            return
        try:
            raw = self._cache_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception:
            return

        now = _utcnow()
        for item in data.values():
            try:
                entry = ResumeCacheEntry(
                    key=item["key"],
                    resume_text=item["resume_text"],
                    summary=item.get("summary", ""),
                    match_rate=float(item.get("match_rate", 0.0)),
                    created_at=datetime.fromisoformat(item["created_at"]),
                    expires_at=datetime.fromisoformat(item["expires_at"]),
                    metadata=item.get("metadata", {}),
                )
            except Exception:
                continue
            if not entry.is_expired(now):
                self._entries[entry.key] = entry
        self._evict_if_needed()

    def _persist_to_disk(self) -> None:
        if not self._cache_path:
            return
        serialized: dict[str, Any] = {}
        for key, entry in self._entries.items():
            serialized[key] = {
                "key": entry.key,
                "resume_text": entry.resume_text,
                "summary": entry.summary,
                "match_rate": entry.match_rate,
                "created_at": entry.created_at.isoformat(),
                "expires_at": entry.expires_at.isoformat(),
                "metadata": entry.metadata,
            }
        tmp_path = self._cache_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(serialized, ensure_ascii=True), encoding="utf-8")
        os.replace(tmp_path, self._cache_path)

    def stats(self) -> dict[str, Any]:
        return {
            "entries": len(self._entries),
            "max_entries": self._max_entries,
            "ttl_seconds": int(self._ttl.total_seconds()),
            "cache_path": str(self._cache_path) if self._cache_path else None,
        }


_resume_cache: ResumeCache | None = None


def get_resume_cache(
    max_entries: int,
    ttl_seconds: int,
    cache_path: str | None,
) -> ResumeCache:
    global _resume_cache
    if _resume_cache is None:
        _resume_cache = ResumeCache(
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
            cache_path=cache_path,
        )
    return _resume_cache
