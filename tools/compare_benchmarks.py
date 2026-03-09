"""Compare two saved benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def by_name(payload: dict) -> dict[str, dict]:
    return {item["name"]: item for item in payload.get("benchmarks", [])}


def pct_delta(before: float | None, after: float | None) -> str:
    if before in (None, 0) or after is None:
        return "n/a"
    delta = ((after - before) / before) * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.2f}%"


def fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}ms"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two benchmark result files.")
    parser.add_argument("before", help="Path to baseline JSON")
    parser.add_argument("after", help="Path to comparison JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    before = load(Path(args.before).resolve())
    after = load(Path(args.after).resolve())

    before_map = by_name(before)
    after_map = by_name(after)
    all_names = sorted(set(before_map) | set(after_map))

    print(f"Before: {args.before}")
    print(f"After : {args.after}")
    print("")

    for name in all_names:
        left = before_map.get(name)
        right = after_map.get(name)
        if not left or not right:
            print(f"- {name}: missing in one side")
            continue

        print(f"- {name}")
        for metric in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"):
            b = left.get(metric)
            a = right.get(metric)
            print(f"  {metric}: {fmt(b)} -> {fmt(a)} ({pct_delta(b, a)})")
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
