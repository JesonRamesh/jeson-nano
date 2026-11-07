"""Decision trace (spec §8.5). No action without evidence: `log()` raises if evidence is missing."""

from __future__ import annotations

from typing import Any

from sgr.contracts import Action, Evidence, TraceRow


class TraceError(ValueError):
    """Raised when a decision is logged without evidence."""


class Trace:
    def __init__(self) -> None:
        self._next = 1
        self._pending: list[TraceRow] = []
        self._last: dict[tuple[str, str], Any] = {}  # (drone, key) -> last decision value
        self.rows: list[TraceRow] = []

    def log(self, t: float, drone: str, action: Action, reason: str, evidence: Evidence | dict | None) -> TraceRow:
        if evidence is None:
            raise TraceError(f"{action} by {drone}: no action without evidence")
        if isinstance(evidence, dict):
            if evidence.get("value") is None:
                raise TraceError(f"{action} by {drone}: evidence.value is missing")
            evidence = Evidence(**evidence)
        if not isinstance(evidence, Evidence):
            raise TraceError(f"{action} by {drone}: evidence must be an Evidence, got {type(evidence).__name__}")
        row = TraceRow(id=f"tr{self._next:05d}", t=round(t, 2), drone=drone, action=action, reason=reason,
                       evidence=evidence)
        self._next += 1
        self._pending.append(row)
        self.rows.append(row)
        return row

    def log_change(self, key: str, value: Any, t: float, drone: str, action: Action, reason: str,
                   evidence: Evidence | dict | None) -> TraceRow | None:
        """Log only when the decision `value` for (drone, key) changes (§8.5: decision changes only)."""
        k = (drone, key)
        if k in self._last and self._last[k] == value:
            return None
        self._last[k] = value
        return self.log(t, drone, action, reason, evidence)

    def drain(self) -> list[TraceRow]:
        out, self._pending = self._pending, []
        return out
