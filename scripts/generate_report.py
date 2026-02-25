import sys
import os
import csv
from datetime import datetime

def main():
   
    if len(sys.argv) < 2:
        raise ValueError("Input CSV path not provided")

    input_csv = sys.argv[1]

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    
    output_dir = "sandbox/runtime"
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "report.csv")

    total_revenue = 0
    rows = []

    with open(input_csv, newline="") as f:
        reader = csv.DictReader(f)

        if "revenue" not in reader.fieldnames:
            raise ValueError("Input CSV missing 'revenue' column")

        for row in reader:
            revenue = int(row["revenue"])
            total_revenue += revenue
            rows.append(row)

    if not rows:
        raise ValueError("No data found in input CSV")

    average_revenue = total_revenue / len(rows)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_days", len(rows)])
        writer.writerow(["total_revenue", total_revenue])
        writer.writerow(["average_revenue", round(average_revenue, 2)])
        writer.writerow(["generated_at", datetime.utcnow().isoformat()])

    print(f"Report generated successfully: {output_csv}")

if __name__ == "__main__":
    main()