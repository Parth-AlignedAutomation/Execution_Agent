"""
scripts/generate_report.py
──────────────────────────
Reads the raw sales CSV produced by the DB executor and writes a
summary report CSV.

Usage (called by the script_executor sub-agent):
    python scripts/generate_report.py <input_csv> <output_csv>
"""
import csv
import sys
from datetime import datetime


def generate(input_path: str, output_path: str) -> None:
    total_revenue = 0.0
    rows = []

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rev = float(row.get("revenue", 0))
            except (ValueError, TypeError):
                rev = 0.0
            total_revenue += rev
            rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_rows",    len(rows)])
        writer.writerow(["total_revenue", round(total_revenue, 2)])
        writer.writerow(["generated_at",  datetime.utcnow().isoformat()])

    print(f"Report written to {output_path}  ({len(rows)} rows, revenue={total_revenue:.2f})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_report.py <input.csv> <output.csv>")
        sys.exit(1)

    generate(sys.argv[1], sys.argv[2])