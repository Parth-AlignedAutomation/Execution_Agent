import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

PROJECT_ROOT = Path(__file__).parent.parent.parent

sys.path.append(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")


def _print_result(client_id: str, steps: list, description: str, final: dict ):
    print("\n" + "=" * 60)
    print(f"  Usecase  : usecase3 — Expense Report → OneDrive")
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
    parser = argparse.ArgumentParser(
        description="Usecase3 — Monthly Expense Report → OneDrive Upload"
    )
    parser.add_argument(
        "--instruction",
        required=True,
        help='e.g. "Upload monthly expense report to OneDrive"',
    )
    parser.add_argument(
        "--visualise",
        action="store_true",
        help="Print LangGraph Mermaid diagram and exit.",
    )
    args = parser.parse_args()

    from planner_agent.planner_agent import planner_graph

    print(f"\n[Usecase3] Instruction: '{args.instruction}'")

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