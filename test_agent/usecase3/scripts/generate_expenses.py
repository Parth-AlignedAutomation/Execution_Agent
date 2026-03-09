import csv
import os
import sys
import random
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

CATEGORIES = {
    "Travel":        [("Flight to Delhi",     8500), ("Train - Mumbai to Pune", 650),
                      ("Cab - Airport",        850),  ("Hotel - 2 nights",     4200)],
    "Meals":         [("Team lunch",          1200),  ("Client dinner",        3400),
                      ("Office snacks",        450),   ("Coffee - client meet",  380)],
    "Office":        [("Printer cartridges",  1100),  ("Stationery",            320),
                      ("Desk organiser",       780),   ("USB hub",               990)],
    "Software":      [("Zoom subscription",  1499),   ("Notion Pro",            800),
                      ("GitHub Copilot",      900),    ("Figma seat",           1200)],
    "Marketing":     [("LinkedIn ads",        5000),  ("Banner printing",      2200),
                      ("Stock photos",         600),   ("SEO tool",             1800)],
    "Miscellaneous": [("Courier charges",      350),  ("Parking fee",           180),
                      ("Postage",              120),   ("Mobile recharge",       599)],
}

APPROVERS = ["Riya Sharma", "Amit Patel", "Neha Singh", "Rohit Verma"]


def generate_expenses(num_rows: int = 30) -> list:
    rows = []
    base_date = datetime(2026, 2, 1)

    for i in range(num_rows):
        category    = random.choice(list(CATEGORIES.keys()))
        desc, base  = random.choice(CATEGORIES[category])
        # slight random variation in amount
        amount      = round(base * random.uniform(0.85, 1.20), 2)
        date        = base_date + timedelta(days=random.randint(0, 27))
        rows.append({
            "date":        date.strftime("%Y-%m-%d"),
            "category":    category,
            "description": desc,
            "amount":      amount,
            "currency":    "INR",
            "approved_by": random.choice(APPROVERS),
        })

    # sort by date
    rows.sort(key=lambda r: r["date"])
    return rows


def save_csv(output_path: str, rows: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    headers = ["date", "category", "description", "amount", "currency", "approved_by"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Expense CSV saved → %s (%d rows)", output_path, len(rows))


def print_summary(rows: list) -> None:
    total = sum(r["amount"] for r in rows)
    by_cat = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + r["amount"]

    print(f"\n  Expense Summary — February 2026")
    print(f"  {'─'*40}")
    for cat, amt in sorted(by_cat.items(), key=lambda x: -x[1]):
        print(f"  {cat:<18}  INR {amt:>10,.2f}")
    print(f"  {'─'*40}")
    print(f"  {'TOTAL':<18}  INR {total:>10,.2f}")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_expenses.py <output.csv>")
        sys.exit(1)

    output_path = sys.argv[1]
    rows        = generate_expenses(num_rows=30)
    save_csv(output_path, rows)
    print_summary(rows)
    print(f"CSV saved → {output_path}")

    