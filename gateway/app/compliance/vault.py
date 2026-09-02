"""Redis-backed token vault — session-safe pseudonymisation.

Replaces the MVP's in-memory request-scoped dict with a concurrency-safe,
multi-tenant store. For each session we keep three Redis structures, all sharing
a TTL so abandoned sessions self-expire:

    vault:{session}:fwd   hash  (label\\x00canonical_value) -> token   # dedupe
    vault:{session}:rev   hash  token -> original_value               # rehydrate
    vault:{session}:cnt   hash  label -> running counter              # token ids

Design (forward/reverse maps + per-label counters, cross-value consistency,
value canonicalisation) is adapted from anon_proxy's PIIStore (MIT), re-homed
onto Redis. The encrypt-at-rest and format-preserving *concepts* are borrowed
from cloakpipe (MIT).

Allocation runs inside a single Lua script so "look up existing, else bump
counter and create" is atomic on the server. The script handles a *batch* of
values in one round trip — masking N entities costs one network call to Redis,
not N (the main latency win for a remote vault like Upstash).

All Redis access is wrapped so a vault outage raises ``VaultUnavailable``, which
the API turns into a clear 503 rather than leaking raw tokens to the client.
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

from redis.exceptions import RedisError

if TYPE_CHECKING:
    from redis.asyncio import Redis

_WHITESPACE = re.compile(r"\s+")


class VaultUnavailable(RuntimeError):
    """Raised when the Redis vault can't be reached / times out."""


# Batched allocator. KEYS = fwd, rev, cnt. ARGV[1] = ttl, then repeating
# 4-tuples: (field, label, original, supplied_token). Returns a token per tuple.
_ALLOC_BATCH_LUA = """
local fwd, rev, cnt = KEYS[1], KEYS[2], KEYS[3]
local ttl = ARGV[1]
local out = {}
local i = 2
while i <= #ARGV do
  local field    = ARGV[i]
  local label    = ARGV[i + 1]
  local original = ARGV[i + 2]
  local supplied = ARGV[i + 3]
  i = i + 4
  local existing = redis.call('HGET', fwd, field)
  local token
  if existing then
    token = existing
  else
    local idx = redis.call('HINCRBY', cnt, label, 1)
    if supplied == '' then
      token = '<' .. label .. '_' .. idx .. '>'
    else
      token = supplied
      if redis.call('HEXISTS', rev, token) == 1 then
        token = supplied .. '_' .. idx
      end
    end
    redis.call('HSET', fwd, field, token)
    redis.call('HSET', rev, token, original)
  end
  out[#out + 1] = token
end
redis.call('EXPIRE', fwd, ttl)
redis.call('EXPIRE', rev, ttl)
redis.call('EXPIRE', cnt, ttl)
return out
"""


def normalize_label(label: str) -> str:
    """Canonical label form: drop a leading ``private_`` and uppercase.

    adapted from anon_proxy/anon_proxy/mapping.py (MIT)
    """
    trimmed = label[len("private_") :] if label.startswith("private_") else label
    return trimmed.upper()


def _canonical(value: str) -> str:
    """Fold whitespace + case so "John  Doe" and "john doe" share a token."""
    return _WHITESPACE.sub(" ", value).strip().casefold()


class RedisVault:
    """Async, session-scoped pseudonymisation vault."""

    def __init__(
        self,
        redis: "Redis",
        *,
        ttl_seconds: int = 3600,
        token_style: str = "placeholder",
        encryption_key: str | None = None,
    ) -> None:
        self._redis = redis
        self._ttl = ttl_seconds
        self._token_style = token_style
        self._script = redis.register_script(_ALLOC_BATCH_LUA)
        self._fernet = self._make_fernet(encryption_key)

    # ---- key helpers ----------------------------------------------------
    @staticmethod
    def _keys(session_id: str) -> tuple[str, str, str]:
        base = f"vault:{session_id}"
        return f"{base}:fwd", f"{base}:rev", f"{base}:cnt"

    @staticmethod
    def _field(label: str, value: str) -> str:
        # NUL separates label from value so neither can spoof the boundary.
        return f"{label}\x00{_canonical(value)}"

    # ---- encryption (on when a key is configured) -----------------------
    @staticmethod
    def _make_fernet(key: str | None):
        if not key:
            return None
        from cryptography.fernet import Fernet  # lazy: only when enabled

        return Fernet(key)

    def _enc(self, value: str) -> str:
        return self._fernet.encrypt(value.encode()).decode() if self._fernet else value

    def _dec(self, value: str | None) -> str | None:
        if value is None or self._fernet is None:
            return value
        return self._fernet.decrypt(value.encode()).decode()

    # ---- format-preserving token candidates -----------------------------
    def _candidate_token(self, label: str, value: str) -> str:
        """Return a supplied token for format-preserving mode, else '' so the
        Lua script falls back to a ``<LABEL_n>`` placeholder."""
        if self._token_style != "format_preserving":
            return ""
        up = label.upper()
        if up in {"EMAIL", "EMAIL_ADDRESS"}:
            return "redacted.user@example.invalid"
        if up in {"PHONE", "PHONE_NUMBER"}:
            digits = re.sub(r"\D", "", value)
            return "+91 55500 00000" if len(digits) >= 10 else "55500"
        return ""  # unknown label -> placeholder

    # ---- public API -----------------------------------------------------
    async def get_or_create_many(
        self, items: list[tuple[str, str]], session_id: str
    ) -> list[str]:
        """Allocate tokens for many (label, value) pairs in ONE round trip.

        Order of returned tokens matches ``items``. Raises ValueError on an
        empty value, VaultUnavailable on a Redis failure.
        """
        if not items:
            return []
        args: list = [self._ttl]
        for label, value in items:
            if not value or not value.strip():
                raise ValueError("RedisVault: value must be non-empty")
            norm = normalize_label(label)
            args.extend(
                [
                    self._field(norm, value),
                    norm,
                    self._enc(value),
                    self._candidate_token(norm, value),
                ]
            )
        try:
            return await self._script(keys=list(self._keys(session_id)), args=args)
        except (RedisError, asyncio.TimeoutError, OSError) as exc:
            raise VaultUnavailable(str(exc)) from exc

    async def get_or_create(self, label: str, value: str, session_id: str) -> str:
        """Single-value convenience wrapper over the batch allocator."""
        tokens = await self.get_or_create_many([(label, value)], session_id)
        return tokens[0]

    async def original(self, token: str, session_id: str) -> str | None:
        """Reverse a single token back to its original value (or None)."""
        _, rev, _ = self._keys(session_id)
        try:
            return self._dec(await self._redis.hget(rev, token))
        except (RedisError, asyncio.TimeoutError, OSError) as exc:
            raise VaultUnavailable(str(exc)) from exc

    async def mapping(self, session_id: str) -> dict[str, str]:
        """Full token -> original map for a session (used by the rehydrator)."""
        _, rev, _ = self._keys(session_id)
        try:
            raw = await self._redis.hgetall(rev)
        except (RedisError, asyncio.TimeoutError, OSError) as exc:
            raise VaultUnavailable(str(exc)) from exc
        return {tok: self._dec(val) for tok, val in raw.items()}

    async def stats(self, session_id: str) -> dict:
        """Non-sensitive counts (safe to expose to the dashboard)."""
        fwd, _, cnt = self._keys(session_id)
        try:
            total = await self._redis.hlen(fwd)
            counters = await self._redis.hgetall(cnt)
        except (RedisError, asyncio.TimeoutError, OSError) as exc:
            raise VaultUnavailable(str(exc)) from exc
        return {"total_mappings": total, "by_label": {k: int(v) for k, v in counters.items()}}

    async def clear(self, session_id: str) -> None:
        try:
            await self._redis.delete(*self._keys(session_id))
        except (RedisError, asyncio.TimeoutError, OSError) as exc:
            raise VaultUnavailable(str(exc)) from exc


class InMemoryVault:
    """Pure-Python in-process vault — same async API as RedisVault.

    Used automatically when Redis is unreachable (e.g. free-tier deploy without
    Upstash configured). Tokens only live for the duration of one request because
    the mapping dict is session-scoped in memory; that is fine for the analyze
    pipeline which masks and rehydrates within the same request.

    No encryption at rest (data never leaves the process), no persistence across
    restarts. Suitable for development and free-tier production where Redis is not
    available.
    """

    def __init__(
        self,
        *,
        token_style: str = "placeholder",
    ) -> None:
        # {session_id: {"fwd": {field: token}, "rev": {token: original}, "cnt": {label: int}}}
        self._store: dict[str, dict] = {}
        self._token_style = token_style

    def _session(self, session_id: str) -> dict:
        if session_id not in self._store:
            self._store[session_id] = {"fwd": {}, "rev": {}, "cnt": {}}
        return self._store[session_id]

    def _candidate_token(self, label: str, value: str) -> str:
        if self._token_style != "format_preserving":
            return ""
        up = label.upper()
        if up in {"EMAIL", "EMAIL_ADDRESS"}:
            return "redacted.user@example.invalid"
        if up in {"PHONE", "PHONE_NUMBER"}:
            digits = re.sub(r"\D", "", value)
            return "+91 55500 00000" if len(digits) >= 10 else "55500"
        return ""

    async def get_or_create_many(
        self, items: list[tuple[str, str]], session_id: str
    ) -> list[str]:
        sess = self._session(session_id)
        fwd, rev, cnt = sess["fwd"], sess["rev"], sess["cnt"]
        tokens: list[str] = []
        for label, value in items:
            if not value or not value.strip():
                raise ValueError("InMemoryVault: value must be non-empty")
            norm = normalize_label(label)
            field = f"{norm}\x00{_canonical(value)}"
            if field in fwd:
                tokens.append(fwd[field])
            else:
                idx = cnt.get(norm, 0) + 1
                cnt[norm] = idx
                supplied = self._candidate_token(norm, value)
                if supplied and supplied not in rev:
                    token = supplied
                elif supplied:
                    token = f"{supplied}_{idx}"
                else:
                    token = f"<{norm}_{idx}>"
                fwd[field] = token
                rev[token] = value
                tokens.append(token)
        return tokens

    async def get_or_create(self, label: str, value: str, session_id: str) -> str:
        tokens = await self.get_or_create_many([(label, value)], session_id)
        return tokens[0]

    async def original(self, token: str, session_id: str) -> str | None:
        return self._session(session_id)["rev"].get(token)

    async def mapping(self, session_id: str) -> dict[str, str]:
        return dict(self._session(session_id)["rev"])

    async def stats(self, session_id: str) -> dict:
        sess = self._session(session_id)
        return {
            "total_mappings": len(sess["fwd"]),
            "by_label": {k: int(v) for k, v in sess["cnt"].items()},
        }

    async def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)
