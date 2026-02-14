"""Static verification for critical data flow paths.

This script verifies (without external APIs/DB) that key call-chain links exist:
- 1-minute ingestion path -> collectors -> DB writes
- 4-hour analysis path -> telegram collection -> report notification
- Telegram notification path -> Bot.send_message/send_photo
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

CHECKS = [
    ("scheduler.py", ["def job_1min_tick", "collector.run()", "funding_collector.run()", "volatility_monitor.run()"]),
    ("collectors/price_collector.py", ["def run(self)", "self.collect_all_prices()", "self.save_to_database(data)", "db.batch_insert_market_data", "db.batch_upsert_cvd_data"]),
    ("collectors/funding_collector.py", ["def run(self)", "self.collect_all_funding_data()", "self.save_to_database(data)", "db.upsert_funding_data"]),
    ("scheduler.py", ["def job_4hour_analysis", "orchestrator.run_scheduled_analysis()"]),
    ("executors/orchestrator.py", ["def run_scheduled_analysis", "telegram_collector.run(hours=4)", "self.run_analysis(symbol, is_emergency=False)"]),
    ("executors/orchestrator.py", ["def node_generate_report", "report_generator.notify(report, chart_bytes=chart_bytes, mode=mode)"]),
    ("executors/report_generator.py", ["def send_telegram_notification", "await bot.send_photo(", "await bot.send_message("]),
]


def main() -> int:
    failures = []

    for rel_path, needles in CHECKS:
        p = ROOT / rel_path
        if not p.exists():
            failures.append(f"Missing file: {rel_path}")
            continue

        text = p.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                failures.append(f"[{rel_path}] missing pattern: {needle}")

    if failures:
        print("DATA FLOW VERIFICATION: FAILED")
        for f in failures:
            print(" -", f)
        return 1

    print("DATA FLOW VERIFICATION: PASSED")
    print(f"Checked {len(CHECKS)} critical call-chain groups.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
