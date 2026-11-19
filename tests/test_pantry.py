from datetime import date

import pytest

from zerodesperdicio.pantry import (
    ExpiryStatus,
    PantryItem,
    PantryState,
    PantryStore,
    assess,
    split_expired,
    urgency,
)

TODAY = date(2026, 11, 10)


def entry_for(expires_on):
    return assess([PantryItem("tomate", expires_on=expires_on)], TODAY)[0]


def test_expired_item():
    e = entry_for(date(2026, 11, 9))
    assert e.status == ExpiryStatus.EXPIRED
    assert e.days_left == -1
    assert e.urgency == 0.0


def test_today_soon_ok_unknown():
    e = entry_for(date(2026, 11, 10))
    assert e.status == ExpiryStatus.TODAY and e.urgency == 1.0
    assert entry_for(date(2026, 11, 13)).status == ExpiryStatus.SOON
    assert entry_for(date(2026, 11, 14)).status == ExpiryStatus.OK
    u = entry_for(None)
    assert u.status == ExpiryStatus.UNKNOWN and u.urgency == 0.0


def test_urgency_values():
    assert urgency(7) == 0.5
    assert urgency(14) == 0.0
    assert urgency(30) == 0.0
    assert urgency(None) == 0.0
    assert urgency(-2) == 0.0


def test_assess_sorts_by_days_left_unknown_last():
    items = [
        PantryItem("a"),
        PantryItem("b", expires_on=date(2026, 11, 13)),
        PantryItem("c", expires_on=date(2026, 11, 9)),
        PantryItem("d", expires_on=date(2026, 11, 10)),
    ]
    assert [e.item.name for e in assess(items, TODAY)] == ["c", "d", "b", "a"]


def test_assess_sets_key():
    assert assess([PantryItem("Tomates maduros")], TODAY)[0].key == "tomate"


def test_split_expired():
    entries = assess([PantryItem("c", expires_on=date(2026, 11, 9)), PantryItem("d")], TODAY)
    usable, expired = split_expired(entries)
    assert [e.item.name for e in usable] == ["d"]
    assert [e.item.name for e in expired] == ["c"]


def test_store_roundtrip(tmp_path):
    store = PantryStore(tmp_path / "p.json")
    state = PantryState(items=[PantryItem("ovo", "6 un", date(2026, 11, 12)), PantryItem("sal")], used=2, discarded=1)
    store.save(state)
    loaded, warning = store.load()
    assert warning is None
    assert loaded == state


def test_store_missing_file(tmp_path):
    loaded, warning = PantryStore(tmp_path / "nao_existe.json").load()
    assert loaded == PantryState(items=[])
    assert warning is None


def test_store_corrupted_file(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("{isso não é json", encoding="utf-8")
    loaded, warning = PantryStore(path).load()
    assert loaded == PantryState(items=[])
    assert warning
