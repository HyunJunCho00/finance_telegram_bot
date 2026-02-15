"""Utilities to avoid storing overlapping OHLC/price fields from Dune rows."""

from __future__ import annotations

from typing import Any

# Existing market_data already stores exchange OHLCV and close prices.
# To avoid duplicated storage, remove direct price/OHLC fields from Dune snapshots.
EXACT_DROP_KEYS = {
    "open",
    "high",
    "low",
    "close",
    "price",
    "eth_price",
    "btc_price",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
}


def _should_drop_key(key: str) -> bool:
    k = key.strip().lower().replace(" ", "_")
    if k in EXACT_DROP_KEYS:
        return True

    # Drop generic OHLC-like variations while keeping non-price signals.
    parts = set(k.split("_"))
    if {"open", "high", "low", "close"}.issubset(parts):
        return True
    if "price" in parts and ("eth" in parts or "btc" in parts or len(parts) <= 2):
        return True
    return False


def sanitize_dune_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    dropped: set[str] = set()
    sanitized: list[dict[str, Any]] = []

    for row in rows:
        clean_row: dict[str, Any] = {}
        for key, value in row.items():
            if _should_drop_key(key):
                dropped.add(key)
                continue
            clean_row[key] = value
        sanitized.append(clean_row)

    return sanitized, sorted(dropped)
