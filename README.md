# Generalised Execution Agent

A modular, LangGraph-powered execution engine that runs any workflow — database queries, script execution, API calls, file operations, and notifications — driven entirely by a planner agent.

---

## How It Works

```
Client Instruction
      │
      ▼
Planner Agent          → decides WHAT steps to run
      │                  (hardcoded now, LLM later)
      │  workflow dict
      ▼
Execution Engine       → builds LangGraph dynamically from planner's steps
      │                  only the steps planner specified are added as nodes
      ▼
Handlers               → each step type is executed by its registered handler
      │
      ├── database_read      → DatabaseHandler  (postgres/mysql/sqlite/mongodb/bigquery)
      ├── script_execution   → ScriptHandler    (python/r/shell/node)
      ├── http_request       → APIHandler       (rest/graphql)
      ├── notification       → NotificationHandler (slack/email/teams)
      ├── file_upload        → FileUploadHandler   (local/s3/gcs)
      └── file_download      → FileDownloadHandler (local/s3/gcs)
```

The execution engine never changes. Only the planner's `plan_node()` changes per usecase.

---

## Project Structure

```
generalised_execution_agent/
│
├── main.py                              ← root entry point
├── requirements.txt
├── .env.example
│
├── execution_agent/                     ← core engine package
│   ├── core/
│   │   └── engine.py                   ← reads planner workflow → builds LangGraph
│   ├── state.py                        ← WorkflowState TypedDict
│   ├── guardrails/
│   │   └── sql_safety.py               ← SQL injection protection (6 rules)
│   ├── policies/
│   │   ├── sql_guardrail_policy.yaml
│   │   └── execution_policy.yaml
│   └── handlers/                       ← all pluggable handlers
│       ├── base_handler.py             ← abstract base class
│       ├── registry.py                 ← maps step types → handler instances
│       ├── database_handler.py         ← thin router → delegates to database/
│       ├── script_handler.py
│       ├── api_handler.py
│       ├── notification_handler.py
│       ├── file_handler.py
│       └── database/                   ← one file per DB engine
│           ├── postgres.py             ← PostgreSQL / NeonDB / Supabase
│           ├── mysql.py                ← MySQL / MariaDB
│           ├── sqlite.py               ← SQLite
│           ├── mongodb.py              ← MongoDB Atlas
│           └── bigquery.py             ← Google BigQuery
│
├── test_agent/                         ← all usecases live here
│   ├── usecase1/                       ← NeonDB + PDF Report
│   └── usecase2/                       ← Weather API fetch
│
├── clients/                            ← debug/testing configs only
│   ├── sales_report/config.yaml
│   ├── api_only/config.yaml
│   └── script_only/config.yaml
│
└── tests/
    └── test_sql_safety_guardrail.py
```

---

## Installation

```bash
# 1. Clone the repo
git clone <repo_url>
cd generalised_execution_agent

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Execution Agent — Core Components

### 1. Engine (`execution_agent/core/engine.py`)

Reads the workflow dict from the planner and dynamically builds a LangGraph pipeline. Only the step types the planner specified are added as nodes — nothing more.

Graph structure built for every workflow:
```
__start__
    │
validation          ← checks all step types have registered handlers
    │
step_0              ← first step planner specified
    │
step_1              ← second step (if any)
    │
step_N              ← Nth step (if any)
    │
audit               ← marks COMPLETED, logs timestamp
    │
__end__

rollback            ← triggered on any failure, deletes created files
```

### 2. State (`execution_agent/state.py`)

Shared TypedDict flowing through every LangGraph node:

| Field | Type | Description |
|---|---|---|
| `workflow` | dict | Planner's full workflow definition |
| `current_step_index` | int | Which step is currently running |
| `files_created` | list | Paths of all files written so far |
| `logs` | list | Audit log messages |
| `status` | str | `INIT` / `RUNNING` / `COMPLETED` / `FAILED` |
| `last_step_output` | str | Human-readable result of last step |
| `error` | str | Error message if something went wrong |

### 3. Handlers (`execution_agent/handlers/`)

Each handler owns one step type. All extend `BaseHandler`:

| Handler | Step Type | Supported Engines |
|---|---|---|
| `DatabaseHandler` | `database_read` | postgres, neon, mysql, sqlite, mongodb, bigquery |
| `ScriptHandler` | `script_execution` | python, r, shell, node |
| `APIHandler` | `http_request` | rest, graphql |
| `NotificationHandler` | `notification` | slack, email, teams |
| `FileUploadHandler` | `file_upload` | local, s3, gcs |
| `FileDownloadHandler` | `file_download` | local, s3, gcs |

### 4. SQL Guardrail (`execution_agent/guardrails/sql_safety.py`)

Automatically applied before any SQL query reaches the database. Rules loaded from `sql_guardrail_policy.yaml`:

| Rule | What It Blocks |
|---|---|
| Strip comments | `SELECT * --DROP TABLE` hidden in comments |
| Block multiple statements | `SELECT 1; DROP TABLE users` |
| Command whitelist | Anything other than `SELECT` and `WITH` |
| Dangerous keywords | `DROP`, `DELETE`, `INSERT`, `UPDATE`, `TRUNCATE` etc. |
| LIMIT required | Queries without a `LIMIT` clause |
| LIMIT max value | `LIMIT` exceeding 1000 rows |

### 5. Registry (`execution_agent/handlers/registry.py`)

Maps step type strings → handler instances. Handlers self-register on import.

```python
registry.get("database_read")    # → DatabaseHandler
registry.available()             # → ['database_read', 'script_execution', ...]
```

---

## Adding a New Database Engine

1. Create `execution_agent/handlers/database/redis.py`
2. Define `connect()`, `execute()`, `fetch()`, `close()`, `ENGINE`, `ALIASES`
3. Add `"redis"` to `_MODULE_NAMES` in `execution_agent/handlers/database/__init__.py`

No other file changes needed.

---

## Adding a New Usecase

1. Create `test_agent/usecaseN/` folder
2. Add `planner_agent/planner_agent.py` — define steps in `plan_node()`
3. Add `main.py` — copy from existing usecase, update description
4. Add `.env` with required keys for that usecase

---
---

# Usecase 1 — NeonDB Sales Report

Fetches sales data from NeonDB, runs a Python script that calls a local Ollama LLM, and produces a styled PDF report.

## Handlers Used

```
Step 1 — database_read      → fetch rows from NeonDB → sales_raw.csv
Step 2 — script_execution   → generate_report.py → sales_report.pdf
```

## Folder Structure

```
test_agent/usecase1/
├── main.py
├── .env
├── planner_agent/
│   └── planner_agent.py        ← database_read + script_execution steps
├── scripts/
│   └── generate_report.py      ← DB fetch + Ollama LLM + PDF generation
└── sandbox/
    └── runtime/
        ├── sales_raw.csv        ← raw data from database
        └── sales_report.pdf     ← AI-generated PDF report
```

## LangGraph Flow

```
validation
    │
step_0  →  DatabaseHandler
    │       engine: postgres
    │       SQL guardrail applied
    │       saves sales_raw.csv
    │
step_1  →  ScriptHandler
    │       runner: python
    │       runs generate_report.py
    │       saves sales_report.pdf
    │
audit
```

## Prerequisites

| Requirement | Details |
|---|---|
| NeonDB | Free at [neon.tech](https://neon.tech) — create `orders` table with `date` and `total_amount` columns |
| Ollama | Install from [ollama.com](https://ollama.com) — run `ollama pull gemma3:4b` |
| Langfuse | Free at [langfuse.com](https://langfuse.com) — create prompt named `sales_report` with label `latest` |

## Environment Variables

```env
# test_agent/usecase1/.env
Neon_URL=postgresql://user:password@host/dbname?sslmode=require
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_PROMPT_NAME=sales_report
LANGFUSE_PROMPT_LABEL=latest
REPORT_SQL_QUERY=SELECT date, SUM(total_amount) AS total_sales FROM orders GROUP BY date LIMIT 100
```

## How to Run

```bash
# Terminal 1 — start Ollama
ollama serve

# Terminal 2 — run the usecase
cd test_agent\usecase1
python main.py --instruction "Generate daily sales report"

# Visualise the LangGraph
python main.py --instruction "Generate daily sales report" --visualise
```

## Expected Output

```
[Planner] Plan ready — 2 steps: ['database_read', 'script_execution']
[Validation] All 2 steps validated.
[SQL Guardrail] Query passed all checks.
[DatabaseHandler] Engine: postgres | Step: fetch_sales_data
[DatabaseHandler] 30 rows saved → sandbox/runtime/sales_raw.csv
[ScriptHandler] Runner: python | Script: scripts/generate_report.py
[PDF saved → sandbox/runtime/sales_report.pdf]

STATUS : COMPLETED
FILES  : ['sandbox/runtime/sales_raw.csv', 'sandbox/runtime/sales_report.pdf']
```

---
---

# Usecase 2 — Weather API Fetch

Fetches current weather data for a city from the OpenWeatherMap REST API and saves the full JSON response. No database, no scripts, no notifications.

## Handlers Used

```
Step 1 — http_request   → GET OpenWeatherMap API → weather_raw.json
```

## Folder Structure

```
test_agent/usecase2/
├── main.py
├── .env
├── planner_agent/
│   └── planner_agent.py        ← http_request step only
└── sandbox/
    └── runtime/
        └── weather_raw.json     ← full API response
```

## LangGraph Flow

```
validation
    │
step_0  →  APIHandler
    │       adapter: rest
    │       GET openweathermap.org/data/2.5/weather
    │       params resolved from .env
    │       saves weather_raw.json
    │
audit
```

## Prerequisites

| Requirement | Details |
|---|---|
| OpenWeatherMap API key | Free at [openweathermap.org/api](https://openweathermap.org/api) |

> **Note:** New API keys take 10–30 minutes to activate after registration.

## Environment Variables

```env
# test_agent/usecase2/.env
OPENWEATHER_API_KEY=your_free_api_key_here
```

## How to Run

```bash
cd test_agent\usecase2
python main.py --instruction "Fetch weather data for Mumbai"

# Visualise the LangGraph
python main.py --instruction "Fetch weather data for Mumbai" --visualise
```

## Expected Output

```
[Planner] Plan ready — 1 steps: ['http_request']
[Validation] All 1 steps validated.
[APIHandler] Adapter: rest | URL: https://api.openweathermap.org/data/2.5/weather
[APIHandler] HTTP 200 → sandbox/runtime/weather_raw.json

STATUS : COMPLETED
FILES  : ['sandbox/runtime/weather_raw.json']
```

## Sample `weather_raw.json`

```json
{
  "name": "Mumbai",
  "main": {
    "temp": 29.5,
    "feels_like": 33.2,
    "humidity": 74
  },
  "weather": [{ "description": "haze" }],
  "wind": { "speed": 4.1 }
}
```

---
---

# Usecase Comparison

| Feature | Usecase 1 | Usecase 2 |
|---|---|---|
| Purpose | Sales report from DB | Weather data from API |
| Handlers | `database_read` + `script_execution` | `http_request` |
| Database | NeonDB (PostgreSQL) | None |
| External API | None | OpenWeatherMap |
| Script | `generate_report.py` | None |
| LLM | Ollama gemma3:4b | None |
| Output | `sales_raw.csv` + `sales_report.pdf` | `weather_raw.json` |
| Steps | 2 | 1 |

---

## Running Tests

```bash
# SQL Guardrail — 25 test cases
python -m pytest tests/test_sql_safety_guardrail.py -v

# Verify all handlers load
python -c "from execution_agent.handlers.registry import load_all_handlers, registry; load_all_handlers(); print(registry.available())"

# Verify DB engines load
python -c "from execution_agent.handlers.database import DB_ENGINES; print(list(DB_ENGINES.keys()))"
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `No handler for step type 'xyz'` | Wrong type name in planner | Use exact names: `database_read`, `script_execution`, `http_request`, `notification`, `file_upload`, `file_download` |
| `connection_url not set` | `.env` missing or wrong key | Key must be exactly `Neon_URL` (capital N, underscore) |
| `401 Unauthorized` (OpenWeatherMap) | API key not yet active | Wait 10–30 min after registration |
| `Cannot connect to Ollama` | Ollama server not running | Run `ollama serve` in a separate terminal |
| `KeyError: 'connect'` | Old `database_handler.py` | Replace with latest version that imports `from execution_agent.handlers.database import DB_ENGINES` |
| `load_all_handlers ImportError` | Old `registry.py` | Replace with latest version that defines `load_all_handlers()` |