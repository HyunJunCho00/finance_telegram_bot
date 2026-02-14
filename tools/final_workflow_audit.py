"""Final static workflow audit.

Purpose:
- Walk critical production workflows without requiring external services.
- Validate that load/runtime safety guards exist in code paths.
- Confirm Telegram dispatch paths are wired.

This script is intentionally static (source inspection) so it can run in CI
or constrained environments where API keys / cloud dependencies are unavailable.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

CHECKS = [
    {
        "name": "1-minute ingestion workflow",
        "file": "scheduler.py",
        "needles": [
            "def job_1min_tick",
            "collector.run()",
            "funding_collector.run()",
            "volatility_monitor.run()",
            "except Exception as e:",
        ],
    },
    {
        "name": "4-hour orchestration entry",
        "file": "scheduler.py",
        "needles": [
            "def job_4hour_analysis",
            "orchestrator.run_scheduled_analysis()",
        ],
    },
    {
        "name": "Scheduled analysis fan-out",
        "file": "executors/orchestrator.py",
        "needles": [
            "def run_scheduled_analysis",
            "telegram_collector.run(hours=4)",
            "for symbol in self.symbols:",
            "self.run_analysis(symbol, is_emergency=False)",
        ],
    },
    {
        "name": "LangGraph + sequential fallback",
        "file": "executors/orchestrator.py",
        "needles": [
            "from langgraph.graph import StateGraph, END",
            "LANGGRAPH_AVAILABLE = True",
            "LANGGRAPH_AVAILABLE = False",
            "return self._run_with_langgraph(symbol, mode, is_emergency)",
            "return self._run_sequential(symbol, mode, is_emergency)",
        ],
    },
    {
        "name": "OOM safety via input trimming",
        "file": "agents/claude_client.py",
        "needles": [
            "def _trim_input",
            "[TRUNCATED",
            "trimmed_message = self._trim_input(user_message, input_cap)",
            "MAX_INPUT_CHARS_",
        ],
    },
    {
        "name": "Chart payload downsize before Judge",
        "file": "executors/orchestrator.py",
        "needles": [
            "chart_bytes_for_vlm = chart_generator.resize_for_low_res(chart_bytes)",
            "chart_image_b64 = chart_generator.chart_to_base64(chart_bytes_for_vlm)",
        ],
    },
    {
        "name": "Telegram report dispatch path",
        "file": "executors/report_generator.py",
        "needles": [
            "async def send_telegram_notification",
            "await bot.send_photo(",
            "await bot.send_message(",
            "caption = message[:1024]",
            "logger.info(\"Telegram notification sent\")",
        ],
    },
    {
        "name": "Interactive Telegram command path",
        "file": "bot/telegram_bot.py",
        "needles": [
            "CommandHandler(\"analyze\", self.cmd_analyze)",
            "orchestrator.run_analysis(symbol, is_emergency=False)",
            "app.run_polling(drop_pending_updates=True)",
        ],
    },
]


def run_checks() -> tuple[list[str], list[str]]:
    passed: list[str] = []
    failed: list[str] = []

    for check in CHECKS:
        rel_path = check["file"]
        target = ROOT / rel_path
        if not target.exists():
            failed.append(f"{check['name']}: missing file {rel_path}")
            continue

        text = target.read_text(encoding="utf-8")
        missing = [needle for needle in check["needles"] if needle not in text]
        if missing:
            failed.append(
                f"{check['name']}: missing {len(missing)} pattern(s) in {rel_path} -> {missing}"
            )
            continue

        passed.append(check["name"])

    return passed, failed


def main() -> int:
    passed, failed = run_checks()

    print("FINAL WORKFLOW AUDIT")
    print(f"  Passed: {len(passed)}")
    print(f"  Failed: {len(failed)}")

    for name in passed:
        print(f"  [PASS] {name}")

    for msg in failed:
        print(f"  [FAIL] {msg}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
