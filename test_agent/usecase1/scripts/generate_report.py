"""
scripts/generate_report.py
──────────────────────────
1. Connects directly to NeonDB and fetches sales data
2. Fetches prompt from Langfuse
3. Calls local Ollama (gemma3:4b) to generate a creative report
4. Saves output as both CSV and a styled PDF

Usage (called by script_executor sub-agent):
    python scripts/generate_report.py <output_csv> <output_pdf>

Environment variables required:
    Neon_URL                – NeonDB connection string
    LANGFUSE_PUBLIC_KEY     – Langfuse public key
    LANGFUSE_SECRET_KEY     – Langfuse secret key
    LANGFUSE_HOST           – e.g. https://cloud.langfuse.com
    LANGFUSE_PROMPT_NAME    – prompt name in Langfuse (default: sales_report)
    LANGFUSE_PROMPT_LABEL   – prompt label (default: latest)
    OLLAMA_HOST             – Ollama server (default: http://localhost:11434)
    OLLAMA_MODEL            – model name (default: gemma3:4b)
"""

import csv
import logging
import os
import re
import sys
from datetime import datetime

import psycopg2
import requests
from dotenv import load_dotenv
from langfuse import Langfuse
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# ── Setup ──────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_HOST          = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL         = os.getenv("OLLAMA_MODEL", "gemma3:4b")
MAX_TOKENS           = int(os.getenv("REPORT_MAX_TOKENS", "2048"))
LANGFUSE_PROMPT_NAME = os.getenv("LANGFUSE_PROMPT_NAME", "sales_report")
LANGFUSE_LABEL       = os.getenv("LANGFUSE_PROMPT_LABEL", "latest")
SQL_QUERY            = os.getenv("REPORT_SQL_QUERY", "SELECT date, revenue FROM sales LIMIT 100")

# ── Brand Colors ───────────────────────────────────────────────────────────────
COLOR_PRIMARY   = colors.HexColor("#1B3A6B")
COLOR_SECONDARY = colors.HexColor("#2E86AB")
COLOR_ACCENT    = colors.HexColor("#F4A261")
COLOR_LIGHT     = colors.HexColor("#F0F4F8")
COLOR_WHITE     = colors.white
COLOR_TEXT      = colors.HexColor("#2D3436")


# ── Step 1: Fetch data from NeonDB ────────────────────────────────────────────

def fetch_data_from_db() -> tuple:
    conn_str = os.getenv("Neon_URL")
    if not conn_str:
        raise RuntimeError("Environment variable 'Neon_URL' is not set.")

    logger.info("Connecting to NeonDB...")
    conn   = psycopg2.connect(conn_str)
    cursor = conn.cursor()

    logger.info("Executing query: %s", SQL_QUERY)
    cursor.execute(SQL_QUERY)

    raw_rows = cursor.fetchall()
    headers  = [desc[0] for desc in cursor.description]

    cursor.close()
    conn.close()

    rows = [dict(zip(headers, row)) for row in raw_rows]
    logger.info("Fetched %d rows from database.", len(rows))
    return headers, rows


# ── Step 2: Save raw data as CSV ──────────────────────────────────────────────

def save_csv(output_path: str, headers: list, rows: list) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV saved to: %s", output_path)


# ── Step 3: Markdown table for prompt ────────────────────────────────────────

def rows_to_markdown_table(headers: list, rows: list) -> str:
    header_row = " | ".join(headers)
    separator  = " | ".join(["---"] * len(headers))
    data_rows  = [
        " | ".join(str(row.get(h, "")) for h in headers)
        for row in rows
    ]
    return "\n".join([header_row, separator] + data_rows)


# ── Step 4: Langfuse prompt ───────────────────────────────────────────────────

def fetch_prompt(langfuse: Langfuse) -> tuple:
    logger.info("Fetching prompt '%s' (label=%s) from Langfuse...", LANGFUSE_PROMPT_NAME, LANGFUSE_LABEL)
    prompt_obj = langfuse.get_prompt(LANGFUSE_PROMPT_NAME, label=LANGFUSE_LABEL)
    logger.info("Fetched prompt version: %s", prompt_obj.version)
    return prompt_obj.prompt, str(prompt_obj.version)


def compile_prompt(template: str, data_table: str, report_date: str) -> str:
    return (
        template
        .replace("{{data_table}}", data_table)
        .replace("{{report_date}}", report_date)
    )


# ── Step 5: Call local Ollama ─────────────────────────────────────────────────

def call_llm(system_prompt: str, user_message: str) -> str:
    """
    Calls the local Ollama server using its OpenAI-compatible
    /v1/chat/completions endpoint.
    Make sure Ollama is running:  ollama serve
    And model is pulled:          ollama pull gemma3:4b
    """
    url = f"{OLLAMA_HOST}/v1/chat/completions"
    logger.info("Calling Ollama at %s — model: %s", url, OLLAMA_MODEL)

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                "max_tokens": MAX_TOKENS,
                "stream":     False,
            },
            timeout=300,   # local models can be slow — 5 min timeout
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_HOST}. "
            "Make sure Ollama is running: run 'ollama serve' in a terminal."
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Ollama returned an error: {e}. "
            f"Make sure the model is pulled: run 'ollama pull {OLLAMA_MODEL}'"
        )


# ── Step 6: Build styled PDF ──────────────────────────────────────────────────

def _build_styles() -> dict:
    return {
        "cover_title": ParagraphStyle(
            "cover_title", fontSize=32, fontName="Helvetica-Bold",
            textColor=COLOR_WHITE, alignment=TA_CENTER, spaceAfter=8,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", fontSize=14, fontName="Helvetica",
            textColor=COLOR_LIGHT, alignment=TA_CENTER, spaceAfter=4,
        ),
        "section_heading": ParagraphStyle(
            "section_heading", fontSize=16, fontName="Helvetica-Bold",
            textColor=COLOR_PRIMARY, spaceBefore=16, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body", fontSize=10, fontName="Helvetica",
            textColor=COLOR_TEXT, leading=16,
            alignment=TA_JUSTIFY, spaceAfter=8,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontSize=10, fontName="Helvetica",
            textColor=COLOR_TEXT, leading=14,
            leftIndent=16, bulletIndent=4, spaceAfter=4,
        ),
        "meta": ParagraphStyle(
            "meta", fontSize=8, fontName="Helvetica",
            textColor=colors.grey, alignment=TA_CENTER,
        ),
        "kpi_label": ParagraphStyle(
            "kpi_label", fontSize=9, fontName="Helvetica-Bold",
            textColor=COLOR_WHITE, alignment=TA_CENTER,
        ),
        "kpi_value": ParagraphStyle(
            "kpi_value", fontSize=18, fontName="Helvetica-Bold",
            textColor=COLOR_ACCENT, alignment=TA_CENTER,
        ),
    }


def _compute_kpis(headers: list, rows: list) -> dict:
    revenue_col = next((h for h in headers if "revenue" in h.lower()), None)
    kpis = {"total_rows": len(rows)}
    if revenue_col:
        values = []
        for r in rows:
            try:
                values.append(float(r[revenue_col]))
            except (ValueError, TypeError):
                pass
        if values:
            kpis["total_revenue"] = sum(values)
            kpis["avg_revenue"]   = sum(values) / len(values)
            kpis["max_revenue"]   = max(values)
            kpis["min_revenue"]   = min(values)
    return kpis


def _make_kpi_table(kpis: dict, styles: dict) -> Table:
    def kpi_cell(label: str, value: str) -> list:
        return [
            Paragraph(label, styles["kpi_label"]),
            Paragraph(value, styles["kpi_value"]),
        ]

    data = [[
        kpi_cell("Total Records", str(kpis.get("total_rows", "-"))),
        kpi_cell("Total Revenue", f"${kpis.get('total_revenue', 0):,.2f}"),
        kpi_cell("Avg Revenue",   f"${kpis.get('avg_revenue',   0):,.2f}"),
        kpi_cell("Peak Revenue",  f"${kpis.get('max_revenue',   0):,.2f}"),
    ]]
    t = Table(data, colWidths=[1.6 * inch] * 4)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), COLOR_PRIMARY),
        ("GRID",          (0, 0), (-1, -1), 0.5, COLOR_SECONDARY),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


def _make_data_table(headers: list, rows: list) -> Table:
    table_data = [headers] + [
        [str(row.get(h, "")) for h in headers]
        for row in rows[:20]
    ]
    col_width = (6.5 * inch) / max(len(headers), 1)
    t = Table(table_data, colWidths=[col_width] * len(headers), repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  COLOR_PRIMARY),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  COLOR_WHITE),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0),  10),
        ("ALIGN",          (0, 0), (-1, 0),  "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COLOR_WHITE, COLOR_LIGHT]),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 1), (-1, -1), 9),
        ("ALIGN",          (0, 1), (-1, -1), "CENTER"),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E0")),
        ("TOPPADDING",     (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 6),
    ]))
    return t


def _parse_llm_sections(report_text: str) -> list:
    """
    Parses LLM markdown into (heading, body) pairs.
    Also strips <think>...</think> tags that some models (like gemma/qwen)
    emit before the actual response.
    """
    # Strip <think> blocks that reasoning models sometimes output
    report_text = re.sub(r"<think>.*?</think>", "", report_text, flags=re.DOTALL).strip()

    sections = []
    current_heading = None
    current_body    = []

    for line in report_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, "\n".join(current_body)))
            current_heading = line.lstrip("# ").strip()
            current_body    = []
        elif line.startswith("# "):
            current_body.append(line.lstrip("# ").strip())
        else:
            current_body.append(line)

    if current_heading is not None:
        sections.append((current_heading, "\n".join(current_body)))

    return sections


def build_pdf(output_path: str, report_text: str, headers: list, rows: list, metadata: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    styles = _build_styles()
    kpis   = _compute_kpis(headers, rows)

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.75*inch,  bottomMargin=0.75*inch,
    )
    story = []

    # Cover block
    cover_data = [
        [Paragraph("SALES PERFORMANCE REPORT", styles["cover_title"])],
        [Paragraph(f"Report Date: {metadata['report_date']}", styles["cover_sub"])],
        [Paragraph(f"Generated by {metadata['model']}", styles["cover_sub"])],
    ]
    cover_table = Table(cover_data, colWidths=[6.5 * inch])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), COLOR_PRIMARY),
        ("TOPPADDING",    (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
        ("LEFTPADDING",   (0, 0), (-1, -1), 20),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 20),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 20))

    # KPI bar
    story.append(_make_kpi_table(kpis, styles))
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_SECONDARY))
    story.append(Spacer(1, 12))

    # LLM sections
    sections = _parse_llm_sections(report_text)
    if sections:
        for heading, body in sections:
            story.append(Paragraph(heading, styles["section_heading"]))
            story.append(HRFlowable(width="40%", thickness=1, color=COLOR_ACCENT, spaceAfter=6))
            for line in body.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith(("- ", "* ", "• ")):
                    clean = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line.lstrip("-*• ").strip())
                    story.append(Paragraph(f"• {clean}", styles["bullet"]))
                else:
                    line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
                    story.append(Paragraph(line, styles["body"]))
            story.append(Spacer(1, 8))
    else:
        for line in report_text.splitlines():
            if line.strip():
                story.append(Paragraph(line.strip(), styles["body"]))

    # Raw data table on new page
    story.append(PageBreak())
    story.append(Paragraph("Raw Data", styles["section_heading"]))
    story.append(HRFlowable(width="40%", thickness=1, color=COLOR_ACCENT, spaceAfter=8))
    story.append(Paragraph(f"Showing {min(20, len(rows))} of {len(rows)} total records.", styles["meta"]))
    story.append(Spacer(1, 8))
    story.append(_make_data_table(headers, rows))

    # Footer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Prompt: {metadata['prompt_name']} v{metadata['prompt_version']}  |  "
        f"Model: {metadata['model']}  |  "
        f"Generated: {metadata['generated_at']}",
        styles["meta"]
    ))

    doc.build(story)
    logger.info("PDF saved to: %s", output_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def generate(output_csv: str, output_pdf: str) -> None:
    # 1. Fetch from DB
    headers, rows = fetch_data_from_db()
    if not rows:
        raise ValueError("No data returned from database.")

    # 2. Save CSV
    save_csv(output_csv, headers, rows)

    # 3. Fetch Langfuse prompt
    langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    prompt_template, prompt_version = fetch_prompt(langfuse)

    # 4. Compile + call LLM
    report_date = datetime.utcnow().strftime("%Y-%m-%d")
    data_table  = rows_to_markdown_table(headers, rows)
    full_prompt = compile_prompt(prompt_template, data_table, report_date)

    system_msg = (
        "You are a senior business analyst. "
        "Produce a clear, creative, and insightful sales report in Markdown format. "
        "Use ## for section headings, bullet points for lists, and **bold** for key figures. "
        "Do not include any <think> tags or internal reasoning — output only the final report."
    )
    report_text = call_llm(system_msg, full_prompt)
    logger.info("LLM response received (%d chars).", len(report_text))

    # 5. Build PDF
    metadata = {
        "report_date":    report_date,
        "generated_at":   datetime.utcnow().isoformat(),
        "model":          OLLAMA_MODEL,
        "prompt_name":    LANGFUSE_PROMPT_NAME,
        "prompt_version": prompt_version,
        "row_count":      len(rows),
    }
    build_pdf(output_pdf, report_text, headers, rows, metadata)

    # 6. Flush Langfuse
    langfuse.flush()

    print(f"CSV  saved → {output_csv}")
    print(f"PDF  saved → {output_pdf}")
    print(f"Rows: {len(rows)} | Model: {OLLAMA_MODEL}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_report.py <output.csv> <output.pdf>")
        sys.exit(1)

    generate(sys.argv[1], sys.argv[2])