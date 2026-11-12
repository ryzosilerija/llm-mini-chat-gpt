"""In-memory conversation history per (npc_id, player_id), with TTL eviction."""

from __future__ import annotations

import threading
import time
from collections import deque


class SessionStore:
    def __init__(self, max_messages: int = 20, ttl_seconds: int = 3600):
        self.max_messages = max_messages
        self.ttl = ttl_seconds
        self._data: dict[tuple[str, str], dict] = {}
        self._lock = threading.Lock()

    def _evict(self):
        now = time.time()
        for key in [k for k, v in self._data.items() if now - v["updated"] > self.ttl]:
            del self._data[key]

    def get(self, npc_id: str, player_id: str) -> list[dict]:
        with self._lock:
            self._evict()
            s = self._data.get((npc_id, player_id))
            return list(s["messages"]) if s else []

    def append(self, npc_id: str, player_id: str, *messages: dict) -> None:
        with self._lock:
            s = self._data.setdefault((npc_id, player_id),
                                      {"messages": deque(maxlen=self.max_messages), "updated": 0.0})
            s["messages"].extend(messages)
            s["updated"] = time.time()

    def reset(self, npc_id: str, player_id: str | None = None) -> int:
        with self._lock:
            keys = [k for k in self._data if k[0] == npc_id and (player_id is None or k[1] == player_id)]
            for k in keys:
                del self._data[k]
            return len(keys)
