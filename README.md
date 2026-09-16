# AI FMCG Sales Analyst

An end-to-end AI-powered sales analytics system for FMCG (fast-moving
consumer goods) data, built as a portfolio project demonstrating data
engineering, statistics, machine learning, forecasting, and LLM/prompt
engineering working together in one coherent architecture.

## Architecture

```
CSV/Excel → Ingestion (load + schema-detect + profile)
          → Data Quality (validate + clean, evidence-logged)
          → Analytics Engine (KPIs, sales trends, products, customers,
             stores, regions, targets, pareto, promotions, anomalies)
          → Forecasting (seasonal-naive baseline + Random Forest, evaluated
             against each other, never presented without a baseline comparison)
          → AI Analyst (Claude interprets validated evidence JSON —
             never raw rows, never invents a number)
          → Analytical QA (evidence completeness gate before anything
             reaches the user)
          → Streamlit (dashboard + natural-language Q&A + management report)
```

**Core design principle:** Python computes every number. Claude only
interprets validated evidence. This is enforced structurally — the
orchestrator builds an evidence object and runs it through a QA check
*before* it's allowed to reach the AI Analyst or the user.

## Why this matters for FMCG specifically

Real FMCG sales exports are messy and inconsistent between systems —
column names vary ("sales" vs "net_sales" vs "turnover"), negative
revenue often represents legitimate credit notes rather than errors,
and demand has strong weekly/seasonal structure that a naive average
would misread. Every module here is schema-adaptive (works whether or
not a target/cost/promotion column exists) and encodes these
domain-specific judgment calls explicitly rather than silently.

## Folder structure

```
ai_fmcg_sales_analyst/
├── app.py                     # Streamlit application
├── requirements.txt
├── data/sample/                # synthetic FMCG dataset for testing/demo
├── src/
│   ├── ingestion/               # loader, schema detection, profiling
│   ├── cleaning/                 # validation checks + safe auto-cleaning
│   ├── analytics/                 # KPI/sales/product/customer/store/region/
│   │                                pareto/promotion/anomaly engines
│   ├── forecasting/                # baseline + Random Forest forecasting
│   ├── ai/                          # Claude client, prompts, intent
│   │                                  routing, orchestrator agent
│   ├── qa/                          # evidence + KPI validation gates
│   └── reporting/                   # Plotly charts, management report
└── tests/                       # 53 tests, all passing
```

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY for the AI Analyst tab
streamlit run app.py
```

Upload any FMCG sales export (CSV/Excel). The app works without an API
key too — every tab except "AI Analyst" is pure deterministic Python.

## Testing

```bash
pytest -v
```

53 tests across ingestion, data quality, every analytics module,
forecasting, QA gates, the orchestrator, and report generation — run
against a generated synthetic 16,700-row FMCG dataset with deliberately
injected messiness (duplicate rows, missing values) so tests prove
real behaviour, not just importability.

## What's deliberately NOT automated

- Negative revenue, cost-exceeding-revenue, and zero-revenue-with-quantity
  rows are flagged, never silently "fixed" — they're business questions,
  not data bugs.
- Forecasts are never shown without a baseline comparison (seasonal-naive),
  so the ML model has to earn its place.
- Claude never receives raw data — only the evidence object the
  orchestrator assembles, gated by the QA evidence check.

## Possible extensions

- SQL-backed ingestion for larger-than-memory datasets
- Price-Volume-Mix decomposition (volume vs. price vs. mix effects)
- Multi-page Streamlit navigation instead of tabs
- Prophet as a third forecasting comparison alongside RF and seasonal-naive
