import argparse
import logging
import sys
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
)


def _print_result(client_id: str, steps: list, description: str, final: dict):
    print("\n" + "=" * 60)
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
    parser = argparse.ArgumentParser(description="Generalized Execution Agent")

    group = parser.add_mutually_exclusive_group(required=True)

    # ── Primary mode — planner driven ─────────────────────────────────────────
    group.add_argument(
        "--instruction",
        help=(
            "Natural language instruction from client.\n"
            "Planner decides what to execute.\n"
            'e.g. "Generate daily sales report"'
        ),
    )

    # ── Debug mode — config driven (bypass planner) ────────────────────────────
    group.add_argument(
        "--debug-client",
        metavar="CLIENT_ID",
        help=(
            "DEBUG ONLY — bypass planner and run config directly.\n"
            "Useful for testing a specific handler in isolation.\n"
            "e.g. sales_report"
        ),
    )

    parser.add_argument(
        "--visualise",
        action="store_true",
        help="Print the LangGraph Mermaid diagram and exit.",
    )

    args = parser.parse_args()

    # ── Mode 1: Instruction → Planner → Execution (correct way) ───────────────
    if args.instruction:
        from planner_agent.planner_agent import planner_graph

        print(f"\n[Planner] Instruction received: '{args.instruction}'")

        if args.visualise:
            print(planner_graph.get_graph().draw_mermaid())
            sys.exit(0)

        result   = planner_graph.invoke({
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

    # ── Mode 2: Config → Execution (debug/testing only) ───────────────────────
    elif args.debug_client:
        from execution_agent.core.engine import build_graph

        print(f"\n[DEBUG] Bypassing planner — loading config for '{args.debug_client}'")
        print("[DEBUG] This mode is for testing handlers only, not for production use.\n")

        compiled_graph, config = build_graph(args.debug_client)

        if args.visualise:
            print(compiled_graph.get_graph().draw_mermaid())
            sys.exit(0)

        initial_state = {
            "workflow":           {"client_id": config["client_id"], "steps": config["steps"]},
            "current_step_index": 0,
            "files_created":      [],
            "logs":               [],
            "status":             "INIT",
            "last_step_output":   None,
            "error":              None,
        }

        final = compiled_graph.invoke(initial_state)
        _print_result(
            client_id   = config["client_id"],
            steps       = [s["type"] for s in config["steps"]],
            description = config.get("description", ""),
            final       = final,
        )


if __name__ == "__main__":
    main()