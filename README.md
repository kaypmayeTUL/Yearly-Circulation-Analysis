# Collection Trend Analysis — Dashboard

A Streamlit app that surfaces collection-usage trends, weeding candidates, and
e-book targets from Alma physical-circulation data. Companion to the
standalone script `generate_library_analysis_fy23_26.py`.

## Files

- `analysis_dashboard.py` — Streamlit app (interactive UI)
- `generate_library_analysis_fy23_26.py` — Standalone script (writes CSVs/PNGs)
- `requirements.txt` — Python dependencies
- `README.md` — This file

## Run locally

```bash
pip install -r requirements.txt
streamlit run analysis_dashboard.py
```

Opens on `http://localhost:8501`. Upload a physical-usage CSV in the sidebar to
begin; add a holdings CSV to unlock the uncirculated-holdings weeding view.

## Input format

**Circulation CSV (required)** — expects one row per Title × Loan Fiscal Year
with these columns:

- `Title`, `Author`, `Publication Year`
- `Call Number`, `Subjects`
- `Loan Fiscal Year` (values like `FY-2023`, `FY-2024`, …)
- `Loans (In House + Not In House)`

**Holdings CSV (optional)** — needs `Call Number` plus either `Title` or
`Title (Normalized)`.

## Tabs

- **Overview** — headline metrics and a region-level trend chart
- **LC Class** — one-letter LC class shifts, filter by discipline
- **LC Subclass** — top movers by subclass, filter by discipline
- **Subject Headings** — full LC heading shifts (not just head terms),
  searchable
- **Geographic** — place-level and region-level trends
- **Weeding** — tiered decay candidates (Strong / Medium / Weak) with
  publication-year protection
- **E-Book Candidates** — top-demand titles with strategic rationale
- **Holdings Weeding** — appears only when a holdings CSV is uploaded;
  shows uncirculated % by LC subclass

## Metric

Every "% change" in this app is a **cumulative trend fit**: a linear regression
through all four fiscal-year loan totals, expressed as a percentage of the
FY-2023 baseline. This means interior spikes and dips register in the shift
number — a subject that went 100 → 200 → 200 → 100 shows near-zero movement
(because it returned to baseline), rather than "0% change" from a naive
endpoint comparison. **R²** (in every trend table) tells you how well the
straight line actually fits the four data points — treat the direction as
reliable when R² is high, and as noisy guidance when R² is near zero.
