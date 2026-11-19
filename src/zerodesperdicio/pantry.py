"""Despensa: itens, status de validade, urgência e persistência local em JSON.

Tudo aqui é determinístico. A IA nunca decide se um alimento está vencido.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path

from .ingredients import ingredient_key

SOON_DAYS = 3
URGENCY_HORIZON_DAYS = 14


class ExpiryStatus(str, Enum):
    EXPIRED = "vencido"
    TODAY = "vence hoje"
    SOON = "vence em breve"
    OK = "dentro da validade"
    UNKNOWN = "sem data"


@dataclass
class PantryItem:
    name: str
    quantity: str = ""
    expires_on: date | None = None


@dataclass(frozen=True)
class PantryEntry:
    item: PantryItem
    key: str
    days_left: int | None
    status: ExpiryStatus
    urgency: float


def urgency(days_left: int | None) -> float:
    """1.0 vence hoje, cai linearmente até 0 em 14 dias. Vencido ou sem data: 0."""
    if days_left is None or days_left < 0:
        return 0.0
    return max(0.0, 1.0 - days_left / URGENCY_HORIZON_DAYS)


def _status(days_left: int | None) -> ExpiryStatus:
    if days_left is None:
        return ExpiryStatus.UNKNOWN
    if days_left < 0:
        return ExpiryStatus.EXPIRED
    if days_left == 0:
        return ExpiryStatus.TODAY
    if days_left <= SOON_DAYS:
        return ExpiryStatus.SOON
    return ExpiryStatus.OK


def assess(items: list[PantryItem], today: date) -> list[PantryEntry]:
    entries = []
    for it in items:
        days = (it.expires_on - today).days if it.expires_on else None
        entries.append(PantryEntry(it, ingredient_key(it.name), days, _status(days), urgency(days)))
    return sorted(entries, key=lambda e: (e.days_left is None, e.days_left if e.days_left is not None else 0))


def split_expired(entries: list[PantryEntry]) -> tuple[list[PantryEntry], list[PantryEntry]]:
    usable = [e for e in entries if e.status != ExpiryStatus.EXPIRED]
    expired = [e for e in entries if e.status == ExpiryStatus.EXPIRED]
    return usable, expired


@dataclass
class PantryState:
    items: list[PantryItem] = field(default_factory=list)
    used: int = 0
    discarded: int = 0


class PantryStore:
    """Guarda a despensa num arquivo JSON local (gravação atômica)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> tuple[PantryState, str | None]:
        if not self.path.exists():
            return PantryState(), None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            items = [
                PantryItem(
                    name=str(i["name"]),
                    quantity=str(i.get("quantity", "")),
                    expires_on=date.fromisoformat(i["expires_on"]) if i.get("expires_on") else None,
                )
                for i in data.get("items", [])
            ]
            return PantryState(items, int(data.get("used", 0)), int(data.get("discarded", 0))), None
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            return PantryState(), f"O arquivo da despensa estava corrompido e foi ignorado ({exc.__class__.__name__})."

    def save(self, state: PantryState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "items": [
                {"name": i.name, "quantity": i.quantity,
                 "expires_on": i.expires_on.isoformat() if i.expires_on else None}
                for i in state.items
            ],
            "used": state.used,
            "discarded": state.discarded,
        }
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)
