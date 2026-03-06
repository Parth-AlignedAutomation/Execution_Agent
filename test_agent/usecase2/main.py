"""
test_agent/usecase2/main.py
────────────────────────────
Usecase2 — Weather data fetch via API + Slack notification.

No scripts. No database. Tests:
    http_request   → APIHandler
    notification   → NotificationHandler

Run from usecase2/ folder:
    cd test_agent/usecase2
    python main.py --instruction "Fetch weather data for Mumbai and notify team"

Requirements in .env:
    OPENWEATHER_API_KEY=your_key_here
    SLACK_WEBHOOK_URL=https://hooks.slack.com/...
"""

import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from usecase2/ folder ───────────────────────────────────────────
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ── Add project root to sys.path so execution_agent package is importable ─────
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
)


def _print_result(client_id: str, steps: list, description: str, final: dict):
    print("\n" + "=" * 60)
    print(f"  Usecase  : usecase2 — Weather API + Slack Notification")
    print(f"  Client   : {client_id}")
    print(f"  Steps    : {steps}")
    print(f"  Desc     : {description}")
    print("=" * 60)
    print(f"  STATUS   : {final.get('status')}")
    print(f"  FILES    : {final.get('files_created', [])}")
    if final.get("error"):
        print(f"  ERROR    : {final['error']}")
    print("  LOGS:")
    for log in final.get("logs", []):
        print(f"    • {log}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Usecase2 — Weather API + Slack Notification")
    parser.add_argument(
        "--instruction",
        required=True,
        help='e.g. "Fetch weather data for Mumbai and notify team"',
    )
    parser.add_argument(
        "--visualise",
        action="store_true",
        help="Print LangGraph Mermaid diagram and exit.",
    )
    args = parser.parse_args()

    from planner_agent.planner_agent import planner_graph

    print(f"\n[Usecase2] Instruction: '{args.instruction}'")

    if args.visualise:
        print(planner_graph.get_graph().draw_mermaid())
        sys.exit(0)

    result = planner_graph.invoke({
        "instruction": args.instruction,
        "workflow":    {},
        "result":      None,
    })

    final    = result["result"]
    workflow = result["workflow"]

    _print_result(
        client_id   = workflow.get("client_id", "unknown"),
        steps       = [s["type"] for s in workflow.get("steps", [])],
        description = args.instruction,
        final       = final,
    )


if __name__ == "__main__":
    main()