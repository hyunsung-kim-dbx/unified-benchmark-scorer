# Unified Benchmark Scorer

A comprehensive benchmark scoring system for comparing performance across Databricks Genie Space and Multi-Agent Supervisor (MAS) systems.

## Overview

This tool evaluates and compares query performance across 4 different systems:

| System | Description | Output Format |
|--------|-------------|---------------|
| Genie Space (Untuned) | Base Databricks Genie Space | SQL + Results |
| Genie Space (Tuned) | Genie Space with sample queries/instructions | SQL + Results |
| MAS + Untuned Genie | Multi-Agent Supervisor using untuned Genie | SQL + Results (benchmark mode) |
| MAS + Tuned Genie | Multi-Agent Supervisor using tuned Genie | SQL + Results (benchmark mode) |

## Features

- **Multi-system comparison** - Run benchmarks against up to 4 systems simultaneously
- **LLM-based result evaluation** - Semantic comparison using Claude or GPT
- **Head-to-head analysis** - Compare tuning impact and MAS vs direct Genie
- **Genie Space discovery** - List and validate available Genie Spaces
- **MAS endpoint management** - Register, health-check, and manage MAS endpoints
- **Benchmark filtering** - Filter by category and difficulty
- **Multiple output formats** - JSON, Markdown, and HTML reports
- **Databricks Apps ready** - On-behalf-of-users authentication

## Quick Start

### Installation

```bash
cd unified_benchmark
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:

```bash
cp .env.example .env
```

Edit with your credentials:

```env
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=dapi...
SQL_WAREHOUSE_ID=your-warehouse-id
GENIE_UNTUNED_SPACE_ID=your-untuned-space-id
GENIE_TUNED_SPACE_ID=your-tuned-space-id
MAS_ENDPOINT=http://localhost:8000
ANTHROPIC_API_KEY=sk-ant-...
```

### Run Streamlit App

```bash
streamlit run app.py
```

### Run CLI Benchmark

```bash
python scripts/run_benchmark.py \
    --benchmarks benchmarks/benchmark.json \
    --genie-untuned-space-id <space-id> \
    --genie-tuned-space-id <space-id> \
    --mas-endpoint http://localhost:8000 \
    --output results/results.json \
    --report results/report.md
```

## Project Structure

```
unified_benchmark/
├── app.py                      # Streamlit frontend
├── app.yaml                    # Databricks Apps config
├── requirements.txt            # Python dependencies
├── api/
│   ├── genie_client.py         # Genie Space client
│   ├── mas_client.py           # MAS benchmark client
│   ├── sql_executor.py         # SQL execution utility
│   ├── workspace_client.py     # Genie Space discovery
│   └── mas_registry.py         # MAS endpoint management
├── scoring/
│   ├── unified_runner.py       # Main benchmark orchestrator
│   ├── llm_judge.py            # LLM-based comparison
│   └── report_generator.py     # Report generation
├── utils/
│   └── auth.py                 # Databricks Apps auth
├── prompts/
│   └── answer_comparison.md    # LLM judge prompt
├── scripts/
│   └── run_benchmark.py        # CLI script
└── benchmarks/
    └── benchmark.json          # Sample benchmarks
```

## Benchmark Format

Benchmarks are JSON arrays with the following schema:

```json
[
  {
    "id": "kpi_001",
    "question": "What is the daily DAU trend?",
    "korean_question": "일별 DAU 추이는 어떻게 되나요?",
    "expected_sql": "SELECT event_date, SUM(active_user) as dau FROM ...",
    "category": "kpi",
    "difficulty": "easy"
  }
]
```

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique benchmark identifier |
| `question` | Yes | Natural language question |
| `expected_sql` | Yes | Ground truth SQL query |
| `korean_question` | No | Korean translation of question |
| `category` | No | Category for filtering (e.g., kpi, social_analytics) |
| `difficulty` | No | Difficulty level (easy, medium, hard) |

## App Tabs

### 🚀 Run Benchmark
- Upload benchmark JSON file or paste directly
- Filter benchmarks by category and difficulty
- Select which systems to test
- Configure timeout and parallelism

### 📈 Results
- View system comparison table
- Head-to-head tuning impact analysis
- Question-level breakdown with filtering
- Download JSON results and Markdown reports

### 🔧 Genie Spaces
- List all accessible Genie Spaces
- Validate Space IDs
- Quick copy for configuration

### 🤖 MAS Endpoints
- Register new MAS endpoints
- Health check endpoints
- Remove inactive endpoints

## Evaluation Criteria

Results are compared using an LLM judge that considers:

- **Data content match** - Actual values match (order may differ)
- **Column equivalence** - Column names may differ but represent same data
- **Numeric tolerance** - Allow small floating-point differences (within 0.01%)
- **Format tolerance** - "1000" vs "1,000" vs "1000.00" are equivalent

Failure categories include:
- `wrong_aggregation` - Incorrect SUM/COUNT/AVG
- `wrong_filter` - Missing or incorrect WHERE conditions
- `wrong_grouping` - Incorrect GROUP BY
- `missing_rows` / `extra_rows` - Row count differences
- `timeout` - System timed out
- `execution_error` - SQL execution failed

## Databricks Apps Deployment

### app.yaml

The app is configured for Databricks Apps deployment with on-behalf-of-users authentication:

```yaml
command:
  - "streamlit"
  - "run"
  - "app.py"
  - "--server.port=8501"
  - "--server.address=0.0.0.0"

env:
  - name: DATABRICKS_HOST
    value: "{{secrets/benchmark-app/databricks-host}}"
  # ... other secrets
```

### Authentication

When deployed as a Databricks App, user identity is forwarded via headers:
- `X-Forwarded-Email` - User's email
- `X-Forwarded-Access-Token` - User's OAuth token
- `X-Forwarded-User` - Username

All SQL queries respect user's data permissions.

## Output Example

### Console Summary

```
================================================================================
                        UNIFIED BENCHMARK RESULTS
================================================================================

System               Score      Passed   Failed   Latency
-------------------- ---------- -------- -------- ------------
Genie (Untuned)      70.0%      35       15       2.5s
Genie (Tuned)        85.0%      42       8        2.3s         ← BEST
MAS + Untuned        68.0%      34       16       4.1s
MAS + Tuned          82.0%      41       9        3.8s

HEAD-TO-HEAD ANALYSIS
---------------------
Tuning Impact (Genie):     +7 improved, -0 regressed (net: +7)
Tuning Impact (MAS):       +8 improved, -1 regressed (net: +7)
MAS vs Genie (Untuned):    -2 questions (MAS slightly worse)
MAS vs Genie (Tuned):      -1 question  (MAS slightly worse)
================================================================================
```

## Requirements

- Python 3.9+
- Databricks workspace with:
  - SQL Warehouse
  - Genie Spaces (untuned and/or tuned)
- MAS endpoint (optional, for MAS testing)
- Anthropic or OpenAI API key (for LLM judge)

## License

Internal use only.

## Contributing

1. Create a feature branch
2. Make changes
3. Test locally with `streamlit run app.py`
4. Submit PR for review
