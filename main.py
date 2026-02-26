import argparse 
import json
import logging
import sys

logging.basicConfig(
    level = logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S"
)

from planner_agent.planner_agent import run_pipeline
from execution_agent.execution import execution_graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Execution Agent Pipeline")

    parser.add_argument(
        "--objective", default="Generate daily sales report",
        help="High-level goal passed to the planner (informational in hardcoded mode).",
    )

    parser.add_argument(
        "--visualise", action="store_true",
        help="Print the LangGraph Mermaid diagram and exit.",
    )
    args = parser.parse_args()


    if args.visualise:
        try:
            print(execution_graph.get_graph().draw_mermaid())
        except Exception as exc:
            print(f"Could not render diagram: {exc}")
        sys.exit(0)


        print(f"\n{'='*60}")
    print(f"  Execution Agent Pipeline")
    print(f"  Objective: {args.objective}")
    print(f"{'='*60}\n")

    final_state = run_pipeline(args.objective)

    print(f"\n{'='*60}")
    print(f"  STATUS  : {final_state['status']}")
    print(f"  FILES   : {final_state['files_created']}")
    if final_state.get("error"):
        print(f"  ERROR   : {final_state['error']}")
    print(f"\n  AUDIT LOG:")
    for entry in final_state.get("logs", []):
        print(f"    • {entry}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()