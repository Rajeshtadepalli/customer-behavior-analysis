# customer-behavior-analysis
 End-to-end customer analytics project — EDA, data cleaning, and machine learning models to predict churn, purchases, and spending behavior. Built with Python, Pandas, and Scikit-learn.
# Customer Behavior Analysis — Phase 1 (T1–T5)

Generic Pandas pipeline that handles Tasks T1–T5 from the project task
breakdown, on **any** customer CSV (no column names are hardcoded — the
script uses keyword heuristics to classify columns).

| Task | What it does | Output files |
|---|---|---|
| T1 | Dataset Structure Exploration | `T1_structure_summary.json`, `T1_sample_rows.csv` |
| T2 | Variable Type Identification (data dictionary) | `T2_data_dictionary.csv` |
| T3 | Missing Value Profiling | `T3_missing_value_report.csv`, `T3_missing_value_heatmap.png`, `T3_missing_value_barplot.png` |
| T4 | Duplicate & Inconsistency Detection | `T4_duplicate_rows.csv`, `T4_inconsistency_report.csv` |
| T5 | Data Cleaning & Preprocessing | `T5_cleaning_log.txt`, `cleaned_customer_data.csv` |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python customer_eda_phase1.py --input path/to/your_customer_data.csv --outdir outputs
```

- `--input` — path to your raw customer CSV file (required)
- `--outdir` — folder where all reports/cleaned data are written (default: `outputs`)

## How column classification works

Each column is heuristically tagged as **Demographic**, **Transaction
History**, **Customer Activity**, **Date/Time**, or **Identifier** based on
keyword matches in its name (e.g. `income`, `purchase`, `login`, `id`).
Semantic data type (Numeric / Categorical / Boolean / Datetime / Text) is
inferred from the values themselves, independent of column naming.

## Cleaning logic applied in T5

- Strips whitespace and standardizes casing on text/categorical columns
- Drops exact duplicate rows
- Fills missing numeric values with the column median
- Fills missing categorical values with the column mode
- Clips unexpected negative values in transaction/activity numeric columns to 0
- Leaves missing datetime values as `NaT`, flagged in the cleaning log for manual review

## Next steps

Phase 2 (feature engineering + predictive modeling) will build on
`cleaned_customer_data.csv` produced here.
